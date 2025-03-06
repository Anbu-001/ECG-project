from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
from flask_cors import CORS
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model("ecg_lstm_model.h5")

# Load ResNet50 for feature extraction
resnet = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
resnet.trainable = False

# Class names
class_names = ["arrhythmia", "chd", "mi", "normal"]

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    features = resnet.predict(img_array, verbose=0)
    features = features.reshape(1, -1, features.shape[-1])
    return features

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided."}), 400

    file = request.files["image"]
    try:
        image = Image.open(file.stream).convert("RGB")
        features = preprocess_image(image)
        prediction = model.predict(features)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction) * 100)

        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/", methods=["GET"])
def home():
    return "✅ ECG Disease Classifier Backend is Running!"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
