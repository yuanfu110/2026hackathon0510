from datetime import datetime
from pathlib import Path
import csv
import io
import uuid

from flask import Flask, jsonify, render_template, request
from PIL import Image


app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset" / "capture"
IMAGE_ROOT = DATASET_DIR / "images" / "train"
LABEL_ROOT = DATASET_DIR / "labels" / "train"
RAW_DIR = DATASET_DIR / "raw"
METADATA_PATH = DATASET_DIR / "metadata.csv"

LABELS = {
    "paper_box": {"name": "紙盒", "folder": "紙盒", "color": "white", "class_id": 0},
    "tableware": {"name": "餐具", "folder": "餐具", "color": "blue", "class_id": 1},
    "aluminum_can": {"name": "鋁罐", "folder": "鋁罐", "color": "red", "class_id": 2},
    "tissue": {"name": "衛生紙", "folder": "衛生紙", "color": "yellow", "class_id": 3},
    "empty": {"name": "空的", "folder": "空的", "color": "black", "class_id": None},
}


def ensure_dataset():
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    LABEL_ROOT.mkdir(parents=True, exist_ok=True)
    for label_key, label in LABELS.items():
        (IMAGE_ROOT / label["folder"]).mkdir(parents=True, exist_ok=True)
        (LABEL_ROOT / label["folder"]).mkdir(parents=True, exist_ok=True)
        (RAW_DIR / label_key).mkdir(parents=True, exist_ok=True)

    data_yaml = DATASET_DIR / "data.yaml"
    data_yaml.write_text(
        "\n".join([
            f"path: {DATASET_DIR.as_posix()}",
            "train: images/train",
            "val: images/train",
            "",
            "names:",
            "  0: 紙盒",
            "  1: 餐具",
            "  2: 鋁罐",
            "  3: 衛生紙",
            "",
        ]),
        encoding="utf-8",
    )

    if not METADATA_PATH.exists():
        with METADATA_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["created_at", "filename", "label_key", "label_name", "class_id"])


def append_metadata(filename, label_key, label):
    with METADATA_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            filename,
            label_key,
            label["name"],
            "" if label["class_id"] is None else label["class_id"],
        ])


def save_yolo_label(label_path, class_id):
    if class_id is None:
        label_path.write_text("", encoding="utf-8")
        return

    # The capture screen labels one centered object per image, so the first
    # training pass uses the full image as a bounding box.
    label_path.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8")


@app.route("/")
def index():
    ensure_dataset()
    return render_template("test_server.html", labels=LABELS)


@app.route("/api/label", methods=["POST"])
def label_image():
    ensure_dataset()

    label_key = request.form.get("label")
    if label_key not in LABELS:
        return jsonify({"success": False, "error": "未知的標籤"}), 400

    if "image" not in request.files:
        return jsonify({"success": False, "error": "沒有收到圖片"}), 400

    label = LABELS[label_key]
    file = request.files["image"]

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        image_path = IMAGE_ROOT / label["folder"] / filename
        raw_path = RAW_DIR / label_key / filename
        label_path = LABEL_ROOT / label["folder"] / f"{Path(filename).stem}.txt"

        image.save(image_path, quality=92)
        image.save(raw_path, quality=92)
        save_yolo_label(label_path, label["class_id"])
        append_metadata(filename, label_key, label)

        return jsonify({
            "success": True,
            "filename": filename,
            "label": label["name"],
            "image_path": str(image_path.relative_to(ROOT)),
            "label_path": str(label_path.relative_to(ROOT)),
        })
    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 500


@app.route("/api/stats")
def stats():
    ensure_dataset()

    counts = {}
    for label in LABELS.values():
        counts[label["name"]] = len(list((IMAGE_ROOT / label["folder"]).glob("*.jpg")))

    return jsonify({"success": True, "counts": counts})


if __name__ == "__main__":
    ensure_dataset()
    app.run(host="0.0.0.0", port=5000, debug=False)
