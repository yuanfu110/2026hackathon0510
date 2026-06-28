from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO
from PIL import Image
import io
import os
import uuid

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "v0.pt")
UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1024:
    raise RuntimeError(
        f"{MODEL_PATH} is not a valid YOLO checkpoint. "
        "Set MODEL_PATH to a real .pt weights file."
    )

model = YOLO(MODEL_PATH)


@app.route("/")
def index():
    return render_template("index.html")


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
            return jsonify({
                "success": True,
                "message": "沒有偵測到物件",
                "detections": []
            })

        best = max(detections, key=lambda x: x["confidence"])

        return jsonify({
            "success": True,
            "main_result": best["class_name"],
            "confidence": best["confidence"],
            "detections": detections,
            "saved_image": filename
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug)
