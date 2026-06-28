from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO
from PIL import Image
import io
import json
import os
import uuid


app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "v0.pt")
UPLOAD_DIR = "uploads"
SAMPLE_PATH = "sample.json"

SAMPLE_BY_CLASS_ID = {
    0: "L",  # 紙盒
    1: "R",  # 餐具
    2: "L",  # 鋁罐
    3: "R",  # 衛生紙
}

sample_state = {"data": ""}

os.makedirs(UPLOAD_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1024:
    raise RuntimeError(
        f"{MODEL_PATH} is not a valid YOLO checkpoint. "
        "Set MODEL_PATH to a real .pt weights file."
    )

model = YOLO(MODEL_PATH)


def update_sample(value):
    sample_state["data"] = value
    with open(SAMPLE_PATH, "w", encoding="utf-8") as file:
        json.dump(sample_state, file, ensure_ascii=False)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/sample")
def sample():
    return jsonify(sample_state)


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
            update_sample("")
            return jsonify({
                "success": True,
                "message": "沒有偵測到物件",
                "detections": [],
                "sample": sample_state
            })

        best = max(detections, key=lambda item: item["confidence"])
        update_sample(SAMPLE_BY_CLASS_ID.get(best["class_id"], ""))

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
