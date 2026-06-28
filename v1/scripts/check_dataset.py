from pathlib import Path
import argparse


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "trash"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_COUNT = 6


def parse_args():
    parser = argparse.ArgumentParser(description="Check YOLO trash dataset structure.")
    parser.add_argument("--dataset", default=str(DATASET), help="Dataset root folder.")
    return parser.parse_args()


def image_files(folder):
    if not folder.exists():
        return []
    return [path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]


def check_split(dataset, split):
    images_dir = dataset / "images" / split
    labels_dir = dataset / "labels" / split
    images = image_files(images_dir)
    errors = []

    if split != "train" and not images_dir.exists() and not labels_dir.exists():
        return None, errors

    if not images_dir.exists():
        errors.append(f"Missing folder: {images_dir}")
    if not labels_dir.exists():
        errors.append(f"Missing folder: {labels_dir}")

    for image_path in images:
        relative_image = image_path.relative_to(images_dir)
        label_path = labels_dir / relative_image.with_suffix(".txt")
        if not label_path.exists():
            errors.append(f"Missing label for {image_path.name}: {label_path}")
            continue

        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{label_path}:{line_number} must have 5 values")
                continue

            try:
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
            except ValueError:
                errors.append(f"{label_path}:{line_number} contains non-numeric values")
                continue

            if class_id < 0 or class_id >= CLASS_COUNT:
                errors.append(f"{label_path}:{line_number} invalid class id {class_id}")
            if any(value < 0 or value > 1 for value in values):
                errors.append(f"{label_path}:{line_number} box values must be between 0 and 1")

    return len(images), errors


def main():
    args = parse_args()
    dataset = Path(args.dataset)
    all_errors = []

    for split in ("train", "val", "test"):
        count, errors = check_split(dataset, split)
        if count is None:
            print(f"{split}: skipped")
            continue
        print(f"{split}: {count} images")
        all_errors.extend(errors)

    if all_errors:
        print("\nDataset errors:")
        for error in all_errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nDataset structure looks valid.")


if __name__ == "__main__":
    main()
