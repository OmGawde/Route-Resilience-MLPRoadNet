import argparse
import os
from pathlib import Path
import shutil
from tqdm import tqdm


def prepare_deepglobe_dataset(input_dir: Path, output_dir: Path):
    """
    Organizes raw DeepGlobe dataset into standard images/ and masks/ folders.
    Raw DeepGlobe format:
      *_sat.jpg -> RGB satellite image
      *_mask.png -> Binary road mask
    """
    img_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(input_dir)
    # Search for all sat images
    sat_files = sorted(list(input_dir.rglob("*_sat.jpg")))

    if not sat_files:
        # Check for regular png/jpg if already renamed
        sat_files = sorted(list(input_dir.rglob("*.jpg")) + list(input_dir.rglob("*.png")))
        print(f"Searching standard format files: found {len(sat_files)}")

    print(f"Found {len(sat_files)} candidate satellite files in {input_dir}")
    paired_count = 0

    for sat_path in tqdm(sat_files, desc="Structuring DeepGlobe dataset"):
        stem = sat_path.name.replace("_sat.jpg", "")
        mask_candidate = sat_path.parent / f"{stem}_mask.png"

        if mask_candidate.exists():
            dest_img = img_dir / f"{stem}.jpg"
            dest_mask = mask_dir / f"{stem}.png"

            if not dest_img.exists():
                shutil.copy2(str(sat_path), str(dest_img))
            if not dest_mask.exists():
                shutil.copy2(str(mask_candidate), str(dest_mask))

            paired_count += 1

    print(f"✅ DeepGlobe Dataset structured successfully! {paired_count} paired samples.")
    print(f"   Images: {img_dir}")
    print(f"   Masks:  {mask_dir}")


def main():
    parser = argparse.ArgumentParser(description="Organize DeepGlobe Road Extraction Dataset")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing downloaded DeepGlobe raw files")
    parser.add_argument("--output-dir", type=str, default="dataset/deepglobe", help="Structured destination directory")
    args = parser.parse_args()

    prepare_deepglobe_dataset(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
