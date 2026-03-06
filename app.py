import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from datetime import datetime
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load model safely
try:
    model = tf.keras.models.load_model("malaria_model.h5")
except:
    model = None
    print("⚠ Model file not found")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = 150

def prepare_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({"error": "Model not loaded"})

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    img = Image.open(filepath).convert("RGB")
    img = prepare_image(img)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        result = "Uninfected"
        confidence = prediction * 100
    else:
        result = "Parasitized"
        confidence = (1 - prediction) * 100

    return jsonify({
        "prediction": result,
        "confidence": round(confidence, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

if __name__ == "__main__":
    app.run(debug=True)