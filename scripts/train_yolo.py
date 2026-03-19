from __future__ import annotations

import os
os.environ["MPLBACKEND"] = "Agg"

import argparse
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolo11m.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs")
    parser.add_argument("--name", type=str, default="football_ball_yolo11m")
    args = parser.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=True,
        single_cls=True,
        degrees=0.0,
        scale=0.2,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        close_mosaic=10,
        patience=15,
        save=True,
        plots=True,
    )


if __name__ == "__main__":
    main()