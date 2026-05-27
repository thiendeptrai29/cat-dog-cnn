from flask import Flask, render_template, request, url_for
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import time
from uuid import uuid4
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("cat_dog_model.h5")

UPLOAD_FOLDER = "static/uploads"
IMAGE_SIZE = (160, 160)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# Predict function

def predict_image(path):

    img = Image.open(path).convert("RGB")

    img = img.resize(IMAGE_SIZE)

    img = np.array(img) / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    confidence = float(prediction[0][0])

    if confidence > 0.5:

        result = "Dog"

        confidence = confidence * 100

    else:

        result = "Cat"

        confidence = (1 - confidence) * 100

    confidence = round(confidence,2)

    return result, confidence

# Home

@app.route("/")

def home():

    return render_template("index.html")

# Predict

@app.route(
    "/predict",
    methods=["POST"]
)

def predict():

    file = request.files["file"]
    filename = secure_filename(file.filename)
    if not filename:
        filename = "upload.jpg"

    unique_filename = f"{uuid4().hex}_{filename}"

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_filename
    )

    file.save(filepath)

    # Time prediction

    start = time.time()

    result, confidence = predict_image(filepath)

    end = time.time()

    predict_time = round(end - start,2)

    image_url = url_for(
        "static",
        filename=f"uploads/{unique_filename}",
        v=int(time.time())
    )

    return render_template(

        "index.html",

        prediction=result,

        confidence=confidence,

        image_path=image_url,

        predict_time=predict_time
    )

if __name__ == "__main__":

    app.run(debug=False, use_reloader=False)
