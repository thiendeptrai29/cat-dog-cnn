from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import uuid

app = Flask(__name__)

# Load AI model
model = tf.keras.models.load_model("cat_dog_model.h5")

# Upload folder
UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Predict function
def predict_image(path):

    img = Image.open(path).convert("RGB")

    img = img.resize((150,150))

    img = np.array(img) / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    confidence = float(prediction[0][0])

    if confidence > 0.5:

        return "🐶 Dog", confidence * 100

    else:

        return "🐱 Cat", (1 - confidence) * 100


# Home page
@app.route("/")
def home():

    return render_template("index.html")


# Predict route
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:

        return render_template(
            "index.html",
            prediction="Không có file!"
        )

    file = request.files["file"]

    if file.filename == "":

        return render_template(
            "index.html",
            prediction="Chưa chọn ảnh!"
        )

    # Random filename
    filename = str(uuid.uuid4()) + ".jpg"

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # Save image
    file.save(filepath)

    # Predict
    result, confidence = predict_image(filepath)

    confidence = round(confidence, 2)

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        image_path=filepath
    )


# Run server
if __name__ == "__main__":

    app.run(debug=True)