from datetime import datetime
from pathlib import Path
import csv
import io
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image


app = Flask(__name__)
PICTURE_REQUEST_ID = 0
latest_picture_state = {
    "image": "",
    "updated_at": "",
}

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
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
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
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


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/picture")
def picture_control():
    ensure_dataset()
    return render_template("picture.html", labels=LABELS)


@app.route("/api/picture-command", methods=["POST"])
def request_picture():
    global PICTURE_REQUEST_ID
    PICTURE_REQUEST_ID += 1
    return jsonify({
        "success": True,
        "request_id": PICTURE_REQUEST_ID,
        "message": "已送出拍照指令，請確認相機頁面已開啟",
    })


@app.route("/api/picture-request")
def picture_request_status():
    return jsonify({
        "success": True,
        "request_id": PICTURE_REQUEST_ID,
    })


@app.route("/api/remote-picture", methods=["POST"])
def save_remote_picture():
    ensure_dataset()

    if "image" not in request.files:
        return jsonify({"success": False, "error": "沒有收到圖片"}), 400

    file = request.files["image"]

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        filename = f"remote_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        image.save(UPLOAD_DIR / filename, quality=92)

        latest_picture_state["image"] = filename
        latest_picture_state["updated_at"] = datetime.now().isoformat(timespec="seconds")

        return jsonify({
            "success": True,
            "image": filename,
            "image_url": f"/uploads/{filename}",
            "updated_at": latest_picture_state["updated_at"],
        })
    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 500


@app.route("/api/latest-picture")
def latest_picture():
    ensure_dataset()

    saved_image = latest_picture_state.get("image") or ""
    image_path = UPLOAD_DIR / saved_image if saved_image else None
    image_exists = bool(saved_image and image_path and image_path.exists())

    return jsonify({
        "success": True,
        "has_image": image_exists,
        "image": saved_image if image_exists else "",
        "image_url": f"/uploads/{saved_image}" if image_exists else "",
        "updated_at": latest_picture_state.get("updated_at", ""),
        "message": "最新遠端拍照" if image_exists else "尚未拍照",
        "main_result": "",
        "confidence": None,
    })


@app.route("/api/label-latest", methods=["POST"])
def label_latest_picture():
    ensure_dataset()

    label_key = request.form.get("label")
    if label_key not in LABELS:
        return jsonify({"success": False, "error": "未知的標籤"}), 400

    saved_image = latest_picture_state.get("image") or ""
    source_path = UPLOAD_DIR / saved_image
    if not saved_image or not source_path.exists():
        return jsonify({"success": False, "error": "目前沒有可標記的照片"}), 400

    label = LABELS[label_key]

    try:
        image = Image.open(source_path).convert("RGB")
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
