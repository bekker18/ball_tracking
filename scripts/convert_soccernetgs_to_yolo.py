from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from tqdm import tqdm


BALL_CATEGORY_ID = 4


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def xywh_to_yolo(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x_center = x + w / 2.0
    y_center = y + h / 2.0
    return x_center / img_w, y_center / img_h, w / img_w, h / img_h


def build_image_lookup(images: List[dict]) -> Dict[str, dict]:
    return {str(img["image_id"]): img for img in images}


def group_ball_annotations(annotations: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for ann in annotations:
        is_ball = ann.get("category_id") == BALL_CATEGORY_ID or ann.get("attributes", {}).get("role") == "ball"
        if not is_ball:
            continue
        image_id = str(ann["image_id"])
        grouped.setdefault(image_id, []).append(ann)
    return grouped


def write_label_file(label_path: Path, yolo_lines: List[str]) -> None:
    label_path.write_text("\n".join(yolo_lines), encoding="utf-8")


def make_yolo_lines_from_annotations(anns: List[dict], img_w: int, img_h: int) -> List[str]:
    lines = []
    for ann in anns:
        bbox = ann.get("bbox_image")
        if bbox is None:
            continue

        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["w"])
        h = float(bbox["h"])

        if w <= 0 or h <= 0:
            continue

        xc, yc, wn, hn = xywh_to_yolo(x, y, w, h, img_w, img_h)
        lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    return lines


def resolve_image_path(clip_dir: Path, img_meta: dict, image_subdir: str) -> Path:
    p = clip_dir / image_subdir / img_meta["file_name"]
    if p.exists():
        return p
    raise FileNotFoundError(f"Image not found: {p}")


def save_image(src_img: Path, dst_img: Path, copy_images: bool) -> None:
    if copy_images:
        shutil.copy2(src_img, dst_img)
    else:
        if not dst_img.exists():
            dst_img.symlink_to(src_img.resolve())


def save_full_frame_sample(
    src_img: Path,
    img_meta: dict,
    anns: List[dict],
    out_img_dir: Path,
    out_lbl_dir: Path,
    stem_prefix: str,
    copy_images: bool,
) -> None:
    unique_stem = f"{stem_prefix}_{Path(img_meta['file_name']).stem}"
    dst_img = out_img_dir / f"{unique_stem}.jpg"
    dst_lbl = out_lbl_dir / f"{unique_stem}.txt"

    save_image(src_img, dst_img, copy_images)

    img_w = int(img_meta["width"])
    img_h = int(img_meta["height"])
    yolo_lines = make_yolo_lines_from_annotations(anns, img_w, img_h)
    write_label_file(dst_lbl, yolo_lines)


def clip_box_to_tile(x: float, y: float, w: float, h: float, tx1: int, ty1: int, tx2: int, ty2: int):
    bx1, by1 = x, y
    bx2, by2 = x + w, y + h

    ix1 = max(bx1, tx1)
    iy1 = max(by1, ty1)
    ix2 = min(bx2, tx2)
    iy2 = min(by2, ty2)

    iw = ix2 - ix1
    ih = iy2 - iy1

    if iw <= 1 or ih <= 1:
        return None

    return ix1 - tx1, iy1 - ty1, iw, ih


def create_tile_labels(
    anns: List[dict],
    tx1: int,
    ty1: int,
    tx2: int,
    ty2: int,
    tile_w: int,
    tile_h: int,
) -> List[str]:
    lines = []

    for ann in anns:
        bbox = ann.get("bbox_image")
        if bbox is None:
            continue

        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["w"])
        h = float(bbox["h"])

        cx = x + w / 2.0
        cy = y + h / 2.0

        # keep only boxes whose center lies inside tile
        if not (tx1 <= cx < tx2 and ty1 <= cy < ty2):
            continue

        clipped = clip_box_to_tile(x, y, w, h, tx1, ty1, tx2, ty2)
        if clipped is None:
            continue

        lx, ly, lw, lh = clipped
        xc, yc, wn, hn = xywh_to_yolo(lx, ly, lw, lh, tile_w, tile_h)
        lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    return lines


def save_tiled_samples(
    src_img: Path,
    img_meta: dict,
    anns: List[dict],
    out_img_dir: Path,
    out_lbl_dir: Path,
    stem_prefix: str,
    tiles_x: int,
    tiles_y: int,
) -> None:
    image = cv2.imread(str(src_img))
    if image is None:
        raise RuntimeError(f"Failed to read image: {src_img}")

    img_h, img_w = image.shape[:2]
    tile_w = img_w // tiles_x
    tile_h = img_h // tiles_y

    base_stem = f"{stem_prefix}_{Path(img_meta['file_name']).stem}"

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x1 = tx * tile_w
            y1 = ty * tile_h
            x2 = img_w if tx == tiles_x - 1 else (tx + 1) * tile_w
            y2 = img_h if ty == tiles_y - 1 else (ty + 1) * tile_h

            tile = image[y1:y2, x1:x2]
            cur_tile_w = x2 - x1
            cur_tile_h = y2 - y1

            yolo_lines = create_tile_labels(
                anns=anns,
                tx1=x1,
                ty1=y1,
                tx2=x2,
                ty2=y2,
                tile_w=cur_tile_w,
                tile_h=cur_tile_h,
            )

            tile_stem = f"{base_stem}_tile_{ty}_{tx}"
            dst_img = out_img_dir / f"{tile_stem}.jpg"
            dst_lbl = out_lbl_dir / f"{tile_stem}.txt"

            cv2.imwrite(str(dst_img), tile)
            write_label_file(dst_lbl, yolo_lines)


def process_clip(
    clip_dir: Path,
    output_root: Path,
    split_name: str,
    image_subdir: str,
    copy_images: bool,
    negative_ratio: float,
    seed: int,
    make_tiles: bool,
    tiles_x: int,
    tiles_y: int,
) -> int:
    json_path = clip_dir / "Labels-GameState.json"
    if not json_path.exists():
        return 0

    data = json.loads(json_path.read_text(encoding="utf-8"))
    images = data["images"]
    annotations = data["annotations"]

    image_lookup = build_image_lookup(images)
    grouped = group_ball_annotations(annotations)

    out_img_dir = output_root / "images" / split_name
    out_lbl_dir = output_root / "labels" / split_name
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    rng = random.Random(seed + hash(clip_dir.name) % 100000)

    all_image_ids = set(image_lookup.keys())
    pos_image_ids = set(grouped.keys())
    neg_image_ids = list(all_image_ids - pos_image_ids)

    if negative_ratio < 1.0:
        keep_n = int(len(neg_image_ids) * negative_ratio)
        keep_n = max(keep_n, 0)
        if keep_n < len(neg_image_ids):
            neg_image_ids = rng.sample(neg_image_ids, keep_n)

    selected_image_ids = sorted(list(pos_image_ids)) + sorted(neg_image_ids)

    written = 0

    for image_id in selected_image_ids:
        img_meta = image_lookup[image_id]
        src_img = resolve_image_path(clip_dir, img_meta, image_subdir)
        anns = grouped.get(image_id, [])

        save_full_frame_sample(
            src_img=src_img,
            img_meta=img_meta,
            anns=anns,
            out_img_dir=out_img_dir,
            out_lbl_dir=out_lbl_dir,
            stem_prefix=clip_dir.name,
            copy_images=copy_images,
        )

        if make_tiles:
            save_tiled_samples(
                src_img=src_img,
                img_meta=img_meta,
                anns=anns,
                out_img_dir=out_img_dir,
                out_lbl_dir=out_lbl_dir,
                stem_prefix=clip_dir.name,
                tiles_x=tiles_x,
                tiles_y=tiles_y,
            )

        written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-subdir", type=str, default="img1")
    parser.add_argument("--copy-images", action="store_true")

    parser.add_argument("--negative-ratio", type=float, default=0.15, help="Fraction of no-ball images to keep")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--make-tiles", action="store_true")
    parser.add_argument("--tiles-x", type=int, default=2)
    parser.add_argument("--tiles-y", type=int, default=2)

    args = parser.parse_args()

    total = 0
    for split_name in ["train", "valid", "test"]:
        split_dir = args.input_root / split_name
        if not split_dir.exists():
            continue

        yolo_split = "val" if split_name == "valid" else split_name
        clip_dirs = [p for p in split_dir.iterdir() if p.is_dir()]

        for clip_dir in tqdm(clip_dirs, desc=f"Processing {split_name}"):
            total += process_clip(
                clip_dir=clip_dir,
                output_root=args.output_root,
                split_name=yolo_split,
                image_subdir=args.image_subdir,
                copy_images=args.copy_images,
                negative_ratio=args.negative_ratio,
                seed=args.seed,
                make_tiles=args.make_tiles,
                tiles_x=args.tiles_x,
                tiles_y=args.tiles_y,
            )

    print(f"Done. Wrote {total} base images (plus tiles if enabled).")


if __name__ == "__main__":
    main()