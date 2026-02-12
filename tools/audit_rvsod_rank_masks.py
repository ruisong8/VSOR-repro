# check if all the img and gt pairs have the same size
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

IMG_EXTS = {".jpg", ".jpeg"}  # your images are jpg
GT_EXT = ".png"

def to_single_channel(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr[..., 0]
    return arr

def bbox_from_binary_mask(binary: np.ndarray):
    ys, xs = np.where(binary > 0)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    # xywh (inclusive -> +1)
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]

def rle_from_binary_mask(binary: np.ndarray):
    rle = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
    return {"size": rle["size"], "counts": rle["counts"]}

def safe_open_size(path: Path):
    with Image.open(path) as im:
        w, h = im.size
    return w, h

def load_gt(path: Path, target_wh=None):
    gt = np.array(Image.open(path))
    gt = to_single_channel(gt)
    if target_wh is not None:
        tw, th = target_wh
        h, w = gt.shape[:2]
        if (w, h) != (tw, th):
            gt = np.array(Image.open(path).resize((tw, th), resample=Image.NEAREST))
            gt = to_single_channel(gt)
    return gt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", required=True, help="RVSOD/train/img")
    ap.add_argument("--gt-root", required=True, help="RVSOD/train/rank saliency masks/img (or ranking ...)")
    ap.add_argument("--out-dir", required=True, help="Where to write audit reports")
    ap.add_argument("--min-area", type=int, default=1, help="Instances smaller than this (pixels) will be flagged")
    ap.add_argument("--max-unique", type=int, default=5000, help="Flag GT with too many unique values (likely not discrete ranks)")
    ap.add_argument("--write-fixed-pkl", action="store_true", help="Also write a fixed train.pkl with GT resized to image size")
    ap.add_argument("--pkl-out", default="train.fixed.pkl", help="Output pkl filename (within out-dir) if write-fixed-pkl")
    args = ap.parse_args()

    img_root = Path(args.img_root)
    gt_root = Path(args.gt_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([p for p in img_root.rglob("*") if p.suffix.lower() in IMG_EXTS])

    images_csv = out_dir / "audit_images.csv"
    inst_csv = out_dir / "audit_instances.csv"
    summary_json = out_dir / "audit_summary.json"

    # Stats
    total_imgs = 0
    missing_gt = 0
    size_mismatch = 0
    flagged_many_unique = 0
    imgs_with_bad_bbox = 0
    imgs_with_oob_bbox = 0
    imgs_with_tiny_inst = 0

    total_instances = 0
    bad_bbox_count = 0
    oob_bbox_count = 0
    tiny_inst_count = 0

    fixed_records = []  # optional

    with images_csv.open("w", newline="") as f_img, inst_csv.open("w", newline="") as f_inst:
        w_img = csv.DictWriter(
            f_img,
            fieldnames=[
                "image_path","gt_path","img_w","img_h","gt_w","gt_h",
                "gt_exists","size_match","sx","sy",
                "unique_count","unique_min","unique_max",
                "flag_many_unique","num_instances",
                "has_bad_bbox","has_oob_bbox","has_tiny_inst",
            ],
        )
        w_img.writeheader()

        w_inst = csv.DictWriter(
            f_inst,
            fieldnames=[
                "image_path","gt_path","img_w","img_h","gt_w","gt_h",
                "rank_value","area","bbox_x","bbox_y","bbox_w","bbox_h",
                "bad_bbox","oob_bbox","tiny_inst",
            ],
        )
        w_inst.writeheader()

        for idx, img_path in enumerate(img_files):
            total_imgs += 1
            rel = img_path.relative_to(img_root)
            gt_rel = rel.with_suffix(GT_EXT)
            gt_path = gt_root / gt_rel

            img_w, img_h = safe_open_size(img_path)

            gt_exists = gt_path.exists()
            if not gt_exists:
                missing_gt += 1
                w_img.writerow({
                    "image_path": str(img_path),
                    "gt_path": str(gt_path),
                    "img_w": img_w, "img_h": img_h,
                    "gt_w": "", "gt_h": "",
                    "gt_exists": False,
                    "size_match": False,
                    "sx": "", "sy": "",
                    "unique_count": "", "unique_min": "", "unique_max": "",
                    "flag_many_unique": False,
                    "num_instances": 0,
                    "has_bad_bbox": False,
                    "has_oob_bbox": False,
                    "has_tiny_inst": False,
                })
                continue

            gt_w, gt_h = safe_open_size(gt_path)
            size_match = (img_w == gt_w and img_h == gt_h)
            sx = (img_w / gt_w) if gt_w else ""
            sy = (img_h / gt_h) if gt_h else ""
            if not size_match:
                size_mismatch += 1

            # Load GT without resizing for auditing mismatch effects
            gt_raw = load_gt(gt_path, target_wh=None)
            gt_raw_h, gt_raw_w = gt_raw.shape[:2]

            u = np.unique(gt_raw)
            unique_count = int(len(u))
            unique_min = int(u.min()) if unique_count else ""
            unique_max = int(u.max()) if unique_count else ""
            flag_many_unique = unique_count > args.max_unique
            if flag_many_unique:
                flagged_many_unique += 1

            # Determine ranks (exclude background 0)
            ranks_vals = u[u != 0]

            num_instances = 0
            has_bad_bbox = False
            has_oob_bbox = False
            has_tiny_inst = False

            # If requested: load GT resized to image size (safe) for pkl generation
            gt_for_pkl = None
            if args.write_fixed_pkl:
                gt_for_pkl = load_gt(gt_path, target_wh=(img_w, img_h))

            annos = []
            ranks_list = []

            for rv in ranks_vals.tolist():
                # Union mask for this rank (do NOT split connected components)
                binary = (gt_raw == rv).astype(np.uint8)
                area = int(binary.sum())
                if area == 0:
                    # shouldn't happen but keep safe
                    continue

                bbox = bbox_from_binary_mask(binary)
                if bbox is None:
                    bad = True
                    oob = True
                    tiny = area < args.min_area
                    has_bad_bbox = True
                    bad_bbox_count += 1
                    if tiny:
                        has_tiny_inst = True
                        tiny_inst_count += 1
                    w_inst.writerow({
                        "image_path": str(img_path),
                        "gt_path": str(gt_path),
                        "img_w": img_w, "img_h": img_h,
                        "gt_w": gt_w, "gt_h": gt_h,
                        "rank_value": int(rv),
                        "area": area,
                        "bbox_x": "", "bbox_y": "", "bbox_w": "", "bbox_h": "",
                        "bad_bbox": True,
                        "oob_bbox": True,
                        "tiny_inst": tiny,
                    })
                    continue

                x, y, w, h = bbox
                bad = not (w > 0 and h > 0)
                # bbox computed in GT coordinate space; if size mismatch, bbox may still appear "in bounds" there.
                # But we also check whether it would be in bounds of the IMAGE size if you naively used it.
                # The strict check: compare against gt_raw size (since bbox comes from gt_raw)
                oob = (x < 0 or y < 0 or (x + w) > gt_raw_w or (y + h) > gt_raw_h)
                tiny = area < args.min_area

                total_instances += 1
                num_instances += 1

                if bad:
                    has_bad_bbox = True
                    bad_bbox_count += 1
                if oob:
                    has_oob_bbox = True
                    oob_bbox_count += 1
                if tiny:
                    has_tiny_inst = True
                    tiny_inst_count += 1

                w_inst.writerow({
                    "image_path": str(img_path),
                    "gt_path": str(gt_path),
                    "img_w": img_w, "img_h": img_h,
                    "gt_w": gt_w, "gt_h": gt_h,
                    "rank_value": int(rv),
                    "area": area,
                    "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
                    "bad_bbox": bad,
                    "oob_bbox": oob,
                    "tiny_inst": tiny,
                })

                # Optional fixed pkl generation (uses resized GT aligned to image)
                if args.write_fixed_pkl and gt_for_pkl is not None:
                    binary2 = (gt_for_pkl == rv).astype(np.uint8)
                    area2 = int(binary2.sum())
                    if area2 < args.min_area:
                        continue
                    bbox2 = bbox_from_binary_mask(binary2)
                    if bbox2 is None:
                        continue

                    annos.append({
                        "bbox": bbox2,           # xywh in IMAGE coordinates now
                        "bbox_mode": 1,          # XYWH_ABS
                        "segmentation": rle_from_binary_mask(binary2),  # compressed RLE
                        "category_id": 0,
                        "iscrowd": 0,
                        "is_person": 0,
                    })
                    ranks_list.append(int(rv))

            if has_bad_bbox:
                imgs_with_bad_bbox += 1
            if has_oob_bbox:
                imgs_with_oob_bbox += 1
            if has_tiny_inst:
                imgs_with_tiny_inst += 1

            w_img.writerow({
                "image_path": str(img_path),
                "gt_path": str(gt_path),
                "img_w": img_w, "img_h": img_h,
                "gt_w": gt_w, "gt_h": gt_h,
                "gt_exists": True,
                "size_match": size_match,
                "sx": sx, "sy": sy,
                "unique_count": unique_count,
                "unique_min": unique_min,
                "unique_max": unique_max,
                "flag_many_unique": flag_many_unique,
                "num_instances": num_instances,
                "has_bad_bbox": has_bad_bbox,
                "has_oob_bbox": has_oob_bbox,
                "has_tiny_inst": has_tiny_inst,
            })

            if args.write_fixed_pkl and len(annos) > 0:
                fixed_records.append({
                    "file_name": str(img_path),
                    "image_id": idx,
                    "height": int(img_h),
                    "width": int(img_w),
                    "annotations": annos,
                    "rank": ranks_list,
                })

    summary = {
        "total_images_scanned": total_imgs,
        "missing_gt_images": missing_gt,
        "size_mismatch_images": size_mismatch,
        "images_flagged_many_unique": flagged_many_unique,
        "images_with_bad_bbox": imgs_with_bad_bbox,
        "images_with_oob_bbox": imgs_with_oob_bbox,
        "images_with_tiny_instance": imgs_with_tiny_inst,
        "total_instances": total_instances,
        "bad_bbox_instances": bad_bbox_count,
        "oob_bbox_instances": oob_bbox_count,
        "tiny_instances": tiny_inst_count,
        "notes": {
            "size_mismatch": "jpg and gt png differ in resolution (potential silent misalignment).",
            "flag_many_unique": "GT seems to have too many unique values -> maybe not discrete rank IDs.",
            "fixed_pkl": "If enabled, GT is resized to image size using NEAREST before generating masks/bboxes.",
        }
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.write_fixed_pkl:
        pkl_path = out_dir / args.pkl_out
        with pkl_path.open("wb") as f:
            import pickle
            pickle.dump(fixed_records, f, protocol=pickle.HIGHEST_PROTOCOL)
        summary["fixed_pkl_path"] = str(pkl_path)
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print("Audit done.")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
