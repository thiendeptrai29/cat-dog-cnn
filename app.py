from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import tensorflow as tf
from flask import Flask, flash, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cat_dog_model.h5"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
DATASET_FOLDER = BASE_DIR / "dataset" / "train"
IMAGE_SIZE = (160, 160)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app = Flask(__name__)
app.config["SECRET_KEY"] = "catdog-cnn-demo"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


def predict_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE)
        batch = np.expand_dims(np.array(image) / 255.0, axis=0)

    score = float(model.predict(batch, verbose=0)[0][0])
    if score > 0.5:
        return "Dog", round(score * 100, 2)
    return "Cat", round((1 - score) * 100, 2)


def get_dataset_stats():
    def image_count(folder):
        if not folder.exists():
            return 0
        return sum(
            1
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower().lstrip(".") in ALLOWED_EXTENSIONS
        )

    cat_count = image_count(DATASET_FOLDER / "cat")
    dog_count = image_count(DATASET_FOLDER / "dogs")
    return {"cat": cat_count, "dog": dog_count, "total": cat_count + dog_count}


def valid_upload(file):
    extension = Path(file.filename or "").suffix.lower().lstrip(".")
    return extension in ALLOWED_EXTENSIONS


@app.errorhandler(413)
def too_large(_error):
    flash("Ảnh vượt quá giới hạn 5 MB.", "error")
    return render_template("predict.html"), 413


@app.route("/")
def home():
    return render_template("index.html", dataset_stats=get_dataset_stats(), evaluation=None)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", result=None)

    file = request.files.get("file")
    if not file or not file.filename:
        flash("Vui lòng chọn một ảnh để dự đoán.", "error")
        return render_template("predict.html", result=None), 400
    if not valid_upload(file):
        flash("Chỉ hỗ trợ ảnh JPG, PNG hoặc WEBP.", "error")
        return render_template("predict.html", result=None), 400

    filename = secure_filename(file.filename) or "upload.jpg"
    unique_filename = f"{uuid4().hex}_{filename}"
    filepath = UPLOAD_FOLDER / unique_filename
    file.save(filepath)

    try:
        label, confidence = predict_image(filepath)
    except (UnidentifiedImageError, OSError):
        filepath.unlink(missing_ok=True)
        flash("Tệp đã tải lên không phải ảnh hợp lệ.", "error")
        return render_template("predict.html", result=None), 400

    result = {
        "label": label,
        "confidence": confidence,
        "image_url": url_for("static", filename=f"uploads/{unique_filename}"),
        "created_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    }
    return render_template("predict.html", result=result)


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
