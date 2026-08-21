import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.osm_ingest import generate_indian_city_dataset, INDIAN_CITIES


def main():
    parser = argparse.ArgumentParser(description="Automated Indian City Satellite + OpenStreetMap Road Ingestor")
    parser.add_argument(
        "--city",
        type=str,
        default="Bengaluru",
        choices=["Bengaluru", "bengaluru", "Delhi", "delhi", "Mumbai", "mumbai", "Hyderabad", "hyderabad"],
        help="Target Indian metropolitan area",
    )
    parser.add_argument("--output-dir", type=str, default="dataset/india_urban", help="Output directory to save dataset")
    parser.add_argument("--num-tiles", type=int, default=30, help="Maximum number of satellite tiles to download")
    parser.add_argument("--zoom", type=int, default=16, help="Tile zoom level (16 = high resolution, ~2.3m/px)")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile width/height in pixels")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"===============================================================")
    print(f"[ISRO / NNRMS Pipeline] Automated Indian Road Asset Mapping")
    print(f"   Target City: {args.city.capitalize()}")
    print(f"   Output Folder: {out_dir}")
    print(f"   Max Tiles: {args.num_tiles} | Zoom: {args.zoom}")
    print(f"===============================================================\n")

    count, img_dir, mask_dir = generate_indian_city_dataset(
        city_name=args.city,
        output_dir=out_dir,
        zoom=args.zoom,
        num_tiles_max=args.num_tiles,
        tile_size=args.tile_size,
    )

    print(f"\n[OK] Pipeline Complete! {count} image-mask pairs are ready.")
    print(f"   Images: {img_dir}")
    print(f"   Masks:  {mask_dir}")
    print(f"\n>> You can now train the model directly by running:")
    print(f"   python scripts/train.py --data-root {out_dir} --output-dir output/india_{args.city.lower()}")


if __name__ == "__main__":
    main()
