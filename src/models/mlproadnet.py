import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Lightweight depthwise-separable convolution with optional dilation."""
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, 3, padding=dilation, dilation=dilation,
            groups=in_ch, bias=False
        )
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class EncoderBlock(nn.Module):
    """Encoder block: two DSD convs + residual shortcut + maxpool."""
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_ch, out_ch, dilation)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch, dilation)
        self.res = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
            if in_ch != out_ch
            else nn.Identity()
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        skip = self.conv2(self.conv1(x)) + self.res(x)
        return self.pool(skip), skip


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        rates = [1, 6, 12, 18]
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            )
            for r in rates
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.GELU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        feats = [b(x) for b in self.branches]
        gp = F.interpolate(self.global_pool(x), size=(h, w), mode="bilinear", align_corners=False)
        feats.append(gp)
        return self.fuse(torch.cat(feats, dim=1))


class MLPMixerLayer(nn.Module):
    """
    MLP-Mixer layer for 2D spatial feature maps.
    Applies spatial token mixing followed by channel mixing.
    """
    def __init__(self, num_patches: int, channels: int, token_mlp_ratio: float = 0.5, channel_mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        hidden_tokens = max(1, int(num_patches * token_mlp_ratio))
        hidden_channels = int(channels * channel_mlp_ratio)

        self.token_mlp = nn.Sequential(
            nn.Linear(num_patches, hidden_tokens),
            nn.GELU(),
            nn.Linear(hidden_tokens, num_patches)
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)  # (B, N, C)

        # Token mixing (spatial)
        y = self.norm1(tokens)
        y = self.token_mlp(y.permute(0, 2, 1)).permute(0, 2, 1)
        tokens = tokens + y

        # Channel mixing
        z = self.norm2(tokens)
        z = self.channel_mlp(z)
        tokens = tokens + z

        return tokens.permute(0, 2, 1).reshape(b, c, h, w)


class MLPMixerBottleneck(nn.Module):
    """Stack of MLP-Mixer layers forming the global reasoning bottleneck."""
    def __init__(self, channels: int, spatial_size: int, depth: int = 4):
        super().__init__()
        num_patches = spatial_size * spatial_size
        self.layers = nn.ModuleList([
            MLPMixerLayer(num_patches, channels) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, max(1, ch // reduction)),
            nn.GELU(),
            nn.Linear(max(1, ch // reduction), ch),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).view(x.shape[0], -1, 1, 1)


class DecoderBlock(nn.Module):
    """Decoder block: bilinear upsampling + skip fusion + DSD conv + channel attention."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_ch + skip_ch, out_ch),
            DepthwiseSeparableConv(out_ch, out_ch),
            ChannelAttention(out_ch)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class MLPRoadNet(nn.Module):
    """
    MLPRoadNet — Hybrid MLP-CNN Architecture for Road Extraction
    Encoder   : DSD (Depthwise-Separable Dilated) blocks
    Bottleneck: ASPP + MLP-Mixer (Global spatial token mixing)
    Decoder   : Dual-Branch (Mask + Centerline) with learned sigmoid gate fusion
    """
    def __init__(self, in_ch: int = 3, base_ch: int = 32, mlp_depth: int = 4, patch_size: int = 512):
        super().__init__()
        c = base_ch

        # Encoder
        self.enc1 = EncoderBlock(in_ch, c, dilation=1)
        self.enc2 = EncoderBlock(c, c * 2, dilation=2)
        self.enc3 = EncoderBlock(c * 2, c * 4, dilation=2)
        self.enc4 = EncoderBlock(c * 4, c * 8, dilation=4)

        # Bottleneck
        bottleneck_spatial = patch_size // 16
        self.aspp = ASPPModule(c * 8, c * 8)
        self.mixer = MLPMixerBottleneck(c * 8, bottleneck_spatial, depth=mlp_depth)
        self.bn_fuse = nn.Sequential(
            nn.Conv2d(c * 8, c * 8, 1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.GELU()
        )

        # Decoder Backbone
        self.dec4 = DecoderBlock(c * 8, c * 8, c * 4)
        self.dec3 = DecoderBlock(c * 4, c * 4, c * 2)
        self.dec2 = DecoderBlock(c * 2, c * 2, c)
        self.dec1 = DecoderBlock(c, c, c)

        # Dual Heads
        self.mask_head = nn.Sequential(
            nn.Conv2d(c, c // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c // 2),
            nn.GELU(),
            nn.Conv2d(c // 2, 1, 1)
        )

        self.cline_head = nn.Sequential(
            nn.Conv2d(c, c // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c // 2),
            nn.GELU(),
            nn.Conv2d(c // 2, 1, 1)
        )

        # Fusion Gate
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(2, 1, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        # Encoder
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        # Bottleneck
        x = self.aspp(x)
        x = self.mixer(x)
        x = self.bn_fuse(x)

        # Decoder
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        # Prediction Heads
        mask_logit = self.mask_head(x)
        cline_logit = self.cline_head(x)

        # Gate Fusion
        gate = self.fusion_gate(torch.cat([mask_logit, cline_logit], dim=1))
        final = gate * mask_logit + (1.0 - gate) * cline_logit

        return final, mask_logit, cline_logit
