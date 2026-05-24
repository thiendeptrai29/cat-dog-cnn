from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load AI model
model = tf.keras.models.load_model("cat_dog_model.h5")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Predict function
def predict_image(path):

    img = Image.open(path).resize((150,150))

    img = np.array(img)/255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        return "🐶 Dog"
    else:
        return "🐱 Cat"

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Predict route
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    result = predict_image(filepath)

    return render_template(
        "index.html",
        prediction=result,
        image_path=filepath
    )

if __name__ == "__main__":
    app.run(debug=True)