from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


BALL_CATEGORY_ID = 4


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def xywh_to_yolo(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x_center = x + w / 2.0
    y_center = y + h / 2.0
    return (
        x_center / img_w,
        y_center / img_h,
        w / img_w,
        h / img_h,
    )


def build_image_lookup(images: List[dict]) -> Dict[str, dict]:
    lookup = {}
    for img in images:
        lookup[str(img["id"])] = img
    return lookup


def group_ball_annotations(annotations: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for ann in annotations:
        is_ball = ann.get("category_id") == BALL_CATEGORY_ID or ann.get("attributes", {}).get("role") == "ball"
        if not is_ball:
            continue
        image_id = str(ann["image_id"])
        grouped.setdefault(image_id, []).append(ann)
    return grouped


def write_label_file(label_path: Path, anns: List[dict], img_w: int, img_h: int) -> None:
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
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        wn = min(max(wn, 0.0), 1.0)
        hn = min(max(hn, 0.0), 1.0)

        # single-class dataset => class id 0
        lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    label_path.write_text("\n".join(lines), encoding="utf-8")


def resolve_image_path(clip_dir: Path, file_name: str | None, image_id: str, image_subdir: str, image_ext: str) -> Path:
    # First try exact file_name from JSON if it exists
    if file_name:
        p = clip_dir / file_name
        if p.exists():
            return p

    # Then try inside image_subdir using image_id
    p = clip_dir / image_subdir / f"{image_id}{image_ext}"
    if p.exists():
        return p

    # Then search recursively
    matches = list(clip_dir.rglob(f"{image_id}{image_ext}"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not resolve image for image_id={image_id} in {clip_dir}")


def process_clip(
    clip_dir: Path,
    output_root: Path,
    split_name: str,
    image_subdir: str,
    image_ext: str,
    copy_images: bool,
) -> int:
    json_path = clip_dir / "Labels-GameState.json"
    if not json_path.exists():
        return 0

    data = json.loads(json_path.read_text(encoding="utf-8"))

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    image_lookup = build_image_lookup(images)
    grouped = group_ball_annotations(annotations)

    out_img_dir = output_root / "images" / split_name
    out_lbl_dir = output_root / "labels" / split_name
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    written = 0

    for image_id, anns in grouped.items():
        img_meta = image_lookup.get(image_id)
        if img_meta is None:
            continue

        img_w = int(img_meta.get("width"))
        img_h = int(img_meta.get("height"))
        file_name = img_meta.get("file_name")

        src_img = resolve_image_path(
            clip_dir=clip_dir,
            file_name=file_name,
            image_id=image_id,
            image_subdir=image_subdir,
            image_ext=image_ext,
        )

        unique_stem = f"{clip_dir.name}_{Path(src_img).stem}"
        dst_img = out_img_dir / f"{unique_stem}{src_img.suffix}"
        dst_lbl = out_lbl_dir / f"{unique_stem}.txt"

        if copy_images:
            shutil.copy2(src_img, dst_img)
        else:
            if not dst_img.exists():
                dst_img.symlink_to(src_img.resolve())

        write_label_file(dst_lbl, anns, img_w, img_h)
        written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True, help="Path to raw SoccerNetGS root")
    parser.add_argument("--output-root", type=Path, required=True, help="Path to YOLO dataset root")
    parser.add_argument("--image-subdir", type=str, default="img1")
    parser.add_argument("--image-ext", type=str, default=".jpg")
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()

    total = 0
    for split_name in ["train", "valid", "test"]:
        split_dir = args.input_root / split_name
        if not split_dir.exists():
            continue

        clip_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        for clip_dir in tqdm(clip_dirs, desc=f"Processing {split_name}"):
            total += process_clip(
                clip_dir=clip_dir,
                output_root=args.output_root,
                split_name="val" if split_name == "valid" else split_name,
                image_subdir=args.image_subdir,
                image_ext=args.image_ext,
                copy_images=args.copy_images,
            )

    print(f"Done. Wrote {total} labeled images.")


if __name__ == "__main__":
    main()