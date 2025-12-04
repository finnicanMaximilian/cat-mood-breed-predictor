import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np

app = Flask(__name__)

# ------------------------
# Load Models
# ------------------------
BREED_MODEL_PATH = "models/breed_model.h5"
MOOD_MODEL_PATH = "models/mood_model.h5"

breed_model = load_model(BREED_MODEL_PATH)
mood_model = load_model(MOOD_MODEL_PATH)

# Labels
BREED_LABELS = [
    "Abyssinian", "Bengal", "Birman", "Bombay",
    "British Shorthair", "Egyptian Mau", "Maine Coon",
    "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx"
]

MOOD_LABELS = [
    "Curious", "Eepy", "Grumpy", "Happy", "Zoomies"
]

# ------------------------
# Helper
# ------------------------
def prepare_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target_size)
    img_array = img_to_array(image)

    # IMPORTANT: Use ResNet50 preprocessing
    img_array = preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ------------------------
# Routes
# ------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file.stream)
    processed_img = prepare_image(img)

    # Predictions
    breed_pred = breed_model.predict(processed_img)[0]
    mood_pred = mood_model.predict(processed_img)[0]

    breed_idx = np.argmax(breed_pred)
    mood_idx = np.argmax(mood_pred)

    result = {
        "breed_prediction": BREED_LABELS[breed_idx],
        "breed_confidence": float(breed_pred[breed_idx]),
        "mood_prediction": MOOD_LABELS[mood_idx],
        "mood_confidence": float(mood_pred[mood_idx])
    }

    return jsonify(result)

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
