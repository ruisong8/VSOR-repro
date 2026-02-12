# check and fix resolution mismatches between img and rank gt.
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg"}
GT_EXT = ".png"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", required=True, help="Dataset/RVSOD/train/img")
    ap.add_argument("--gt-root", required=True, help="Dataset/RVSOD/train/ranking saliency masks/img")
    ap.add_argument("--dry-run", action="store_true", help="Only report, do not modify files")
    ap.add_argument("--limit", type=int, default=0, help="Resize at most N files (0 = no limit)")
    args = ap.parse_args()

    img_root = Path(args.img_root)
    gt_root = Path(args.gt_root)

    img_files = sorted([p for p in img_root.rglob("*") if p.suffix.lower() in IMG_EXTS])

    total = 0
    missing = 0
    mismatch = 0
    fixed = 0

    for img_path in img_files:
        total += 1
        rel = img_path.relative_to(img_root)
        gt_rel = rel.with_suffix(GT_EXT)
        gt_path = gt_root / gt_rel

        if not gt_path.exists():
            missing += 1
            continue

        with Image.open(img_path) as im:
            iw, ih = im.size

        with Image.open(gt_path) as gm:
            gw, gh = gm.size

        if (iw, ih) == (gw, gh):
            continue

        mismatch += 1
        print(f"[mismatch] img={iw}x{ih} gt={gw}x{gh} -> {gt_path}")

        if args.dry_run:
            continue

        if args.limit and fixed >= args.limit:
            continue

        # Resize GT to image size using NEAREST to preserve discrete rank IDs
        with Image.open(gt_path) as gm:
            # keep mode; NEAREST is critical
            gm2 = gm.resize((iw, ih), resample=Image.NEAREST)
            # Overwrite in place
            gm2.save(gt_path)

        fixed += 1

    print("\n=== Summary ===")
    print(f"Images scanned: {total}")
    print(f"Missing GT:     {missing}")
    print(f"Mismatched GT:  {mismatch}")
    print(f"Fixed (wrote):  {fixed}")
    if args.dry_run:
        print("Dry-run mode: no files were modified.")

if __name__ == "__main__":
    main()
