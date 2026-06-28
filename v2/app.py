from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask import send_from_directory
from ultralytics import YOLO
from PIL import Image
from datetime import datetime, timezone
from pathlib import Path
import csv
import io
import json
import os
import time
import uuid


app = Flask(__name__)
PICTURE_REQUEST_ID = 0

MODEL_PATH = os.environ.get("MODEL_PATH", "v3.pt")
UPLOAD_DIR = "uploads"
SAMPLE_PATH = "sample.json"
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset" / "capture"
IMAGE_ROOT = DATASET_DIR / "images" / "train"
LABEL_ROOT = DATASET_DIR / "labels" / "train"
RAW_DIR = DATASET_DIR / "raw"
METADATA_PATH = DATASET_DIR / "metadata.csv"

SAMPLE_BY_CLASS_ID = {
    0: "L",  # 紙盒
    1: "R",  # 餐具
    2: "L",  # 鋁罐
    3: "R",  # 衛生紙
}

LABELS = {
    "paper_box": {"name": "紙盒", "folder": "紙盒", "color": "white", "class_id": 0},
    "tableware": {"name": "餐具", "folder": "餐具", "color": "blue", "class_id": 1},
    "aluminum_can": {"name": "鋁罐", "folder": "鋁罐", "color": "red", "class_id": 2},
    "tissue": {"name": "衛生紙", "folder": "衛生紙", "color": "yellow", "class_id": 3},
    "empty": {"name": "空的", "folder": "空的", "color": "black", "class_id": None},
}

sample_state = {
    "data": "",
    "main_result": "",
    "confidence": None,
    "message": "尚未偵測",
    "detections": [],
    "saved_image": "",
    "updated_at": ""
}

os.makedirs(UPLOAD_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1024:
    raise RuntimeError(
        f"{MODEL_PATH} is not a valid YOLO checkpoint. "
        "Set MODEL_PATH to a real .pt weights file."
    )

model = YOLO(MODEL_PATH)


def load_sample():
    if not os.path.exists(SAMPLE_PATH):
        return

    try:
        with open(SAMPLE_PATH, "r", encoding="utf-8") as file:
            saved_state = json.load(file)

        if isinstance(saved_state, dict):
            sample_state.update(saved_state)
    except (OSError, json.JSONDecodeError):
        pass


def update_sample(value, **extra_state):
    sample_state.update(extra_state)
    sample_state["data"] = value
    sample_state["updated_at"] = datetime.now(timezone.utc).isoformat()

    with open(SAMPLE_PATH, "w", encoding="utf-8") as file:
        json.dump(sample_state, file, ensure_ascii=False)


def ensure_dataset():
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    LABEL_ROOT.mkdir(parents=True, exist_ok=True)
    for label_key, label in LABELS.items():
        (IMAGE_ROOT / label["folder"]).mkdir(parents=True, exist_ok=True)
        (LABEL_ROOT / label["folder"]).mkdir(parents=True, exist_ok=True)
        (RAW_DIR / label_key).mkdir(parents=True, exist_ok=True)

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

    label_path.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8")


load_sample()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/show.html")
@app.route("/show")
def show():
    return render_template("show.html")


@app.route("/sample")
def sample():
    return jsonify({"data": sample_state.get("data", "")})


@app.route("/api/sample-state")
def sample_full_state():
    return jsonify(sample_state)


@app.route("/sample-stream")
def sample_stream():
    def event_stream():
        last_payload = None

        while True:
            payload = json.dumps(sample_state, ensure_ascii=False)

            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload

            time.sleep(0.2)

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/picture")
def picture_control():
    return render_template("picture_display.html")


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


@app.route("/api/latest-picture")
def latest_picture():
    load_sample()
    saved_image = sample_state.get("saved_image") or ""
    image_path = Path(UPLOAD_DIR) / saved_image if saved_image else None
    image_exists = bool(saved_image and image_path and image_path.exists())

    return jsonify({
        "success": True,
        "has_image": image_exists,
        "image": saved_image if image_exists else "",
        "image_url": f"/uploads/{saved_image}" if image_exists else "",
        "updated_at": sample_state.get("updated_at", ""),
        "message": sample_state.get("message", ""),
        "main_result": sample_state.get("main_result", ""),
        "confidence": sample_state.get("confidence"),
    })


@app.route("/api/label-latest", methods=["POST"])
def label_latest_picture():
    ensure_dataset()
    load_sample()

    label_key = request.form.get("label")
    if label_key not in LABELS:
        return jsonify({"success": False, "error": "未知的標籤"}), 400

    saved_image = sample_state.get("saved_image") or ""
    source_path = Path(UPLOAD_DIR) / saved_image
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


@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "沒有收到圖片"
        }), 400

    file = request.files["image"]

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        filename = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(UPLOAD_DIR, filename)
        image.save(save_path)

        results = model(image, conf=0.25)
        detections = []

        for result in results:
            names = result.names

            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "class_id": cls_id,
                    "class_name": names[cls_id],
                    "confidence": round(conf, 3),
                    "box": {
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1)
                    }
                })

        if len(detections) == 0:
            update_sample(
                "",
                main_result="",
                confidence=None,
                message="沒有偵測到物件",
                detections=[],
                saved_image=filename
            )
            return jsonify({
                "success": True,
                "message": "沒有偵測到物件",
                "detections": [],
                "sample": sample_state
            })

        best = max(detections, key=lambda item: item["confidence"])
        update_sample(
            SAMPLE_BY_CLASS_ID.get(best["class_id"], ""),
            main_result=best["class_name"],
            confidence=best["confidence"],
            message="偵測完成",
            detections=detections,
            saved_image=filename
        )

        return jsonify({
            "success": True,
            "main_result": best["class_name"],
            "confidence": best["confidence"],
            "detections": detections,
            "saved_image": filename,
            "sample": sample_state
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug)
