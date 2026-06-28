from pathlib import Path
import argparse
import shutil

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "dataset" / "trash" / "data.yaml"
DEFAULT_MODEL = ROOT / "yolov8n.pt"
DEFAULT_PROJECT = ROOT / "runs" / "trash"
DEFAULT_EXPORT = ROOT / "models" / "trash_best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the trash detection YOLO model.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to YOLO dataset yaml.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Base model or previous weights.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default=None, help="Device, for example 0, cpu, or cuda:0.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--name", default="train", help="Run name under runs/trash.")
    parser.add_argument(
        "--export-best",
        default=str(DEFAULT_EXPORT),
        help="Copy trained best.pt here after training. Use empty string to skip.",
    )
    parser.add_argument(
        "--no-export-best",
        action="store_true",
        help="Do not copy best.pt after training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(DEFAULT_PROJECT),
        "name": args.name,
        "exist_ok": True,
    }
    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"

    print(f"Training output: {save_dir}")
    print(f"Best weights: {best_pt}")

    if args.export_best and not args.no_export_best:
        export_path = Path(args.export_best)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, export_path)
        print(f"Copied best weights to: {export_path}")


if __name__ == "__main__":
    main()
