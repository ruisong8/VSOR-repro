# generate pkl file from RVSOD
import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
from pycocotools import mask as mask_utils

BOXMODE_XYWH_ABS = 1  # detectron2 BoxMode.XYWH_ABS

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def bbox_from_binary_mask(binary: np.ndarray):
    ys, xs = np.where(binary > 0)
    if xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]

def rle_from_binary_mask(binary: np.ndarray):
    rle = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
    # Keep only size + counts to be safe/compact
    return {"size": rle["size"], "counts": rle["counts"]}

def to_single_channel(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr[..., 0]
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", required=True, help="Dataset/RVSOD/train/img")
    ap.add_argument("--gt-root", required=True, help="Dataset/RVSOD/train/ranking saliency masks/img")
    ap.add_argument("--out", required=True, help="Dataset/RVSOD/RVSOD/train.pkl")
    ap.add_argument("--min-area", type=int, default=1, help="filter tiny instances by pixel area")
    args = ap.parse_args()

    img_root = Path(args.img_root)
    gt_root = Path(args.gt_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img_files = sorted([p for p in img_root.rglob("*") if p.suffix.lower() in IMG_EXTS])

    records = []
    missing_gt = 0

    for idx, img_path in enumerate(img_files):
        rel = img_path.relative_to(img_root)
        
        # change suffix from .jpg to .png for GT
        gt_rel = rel.with_suffix(".png")
        
        gt_path = gt_root / gt_rel

        if not gt_path.exists():
            print(gt_path)
            missing_gt += 1
            continue

        # read size from image (fast)
        with Image.open(img_path) as im:
            w, h = im.size

        gt = np.array(Image.open(gt_path))
        gt = to_single_channel(gt)

        # unique ranks excluding background 0
        ranks_vals = np.unique(gt)
        ranks_vals = ranks_vals[ranks_vals != 0]

        annos = []
        raw_rvs = []

        for rv in ranks_vals.tolist():
            # binary = (gt == rv).astype(np.uint8)

            # # connected components: each component is an instance
            # num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
            # # labels: 0 is background
            # for lab in range(1, num_labels):
            #     inst = (labels == lab).astype(np.uint8)
            #     area = int(inst.sum())
            #     if area < args.min_area:
            #         continue

            #     bbox = bbox_from_binary_mask(inst)
            #     if bbox is None:
            #         continue

            #     annos.append({
            #         "bbox": bbox,
            #         "bbox_mode": BOXMODE_XYWH_ABS,
            #         "segmentation": rle_from_binary_mask(inst),
            #         "category_id": 0,
            #         "iscrowd": 0,
            #         "is_person": 0,  # you said not using eye fixation / manual masks -> keep 0
            #     })
            #     ranks.append(int(rv))

            inst = (gt == rv).astype(np.uint8)  # IMPORTANT: do NOT split connected components
            area = int(inst.sum())
            if area < args.min_area:
                continue
            
            bbox = bbox_from_binary_mask(inst)
            if bbox is None:
                continue
            
            annos.append({
                "bbox": bbox,
                "bbox_mode": BOXMODE_XYWH_ABS,
                "segmentation": rle_from_binary_mask(inst),
                "category_id": 0,
                "iscrowd": 0,
                "is_person": 0,
            })
            raw_rvs.append(int(rv))

        num_inst = len(raw_rvs)

        # sort indices by original rank value (ascending)
        sorted_indices = sorted(range(num_inst), key=lambda i: raw_rvs[i])
        
        ranks = [0] * num_inst
        
        for new_rank, idx_in_annos in enumerate(sorted_indices):
            ranks[idx_in_annos] = new_rank

        if len(annos) == 0:
            continue

        records.append({
            "file_name": str(img_path),
            "image_id": idx,
            "height": int(h),
            "width": int(w),
            "annotations": annos,
            "rank": ranks,  # IMPORTANT: top-level, aligned with annos
        })

    with out_path.open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(records)} samples to {out_path}")
    print(f"Images scanned: {len(img_files)}")
    print(f"Missing GT: {missing_gt}")

if __name__ == "__main__":
    main()
