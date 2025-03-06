import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
import os

# Load the trained model
model = tf.keras.models.load_model("ecg_lstm_model.h5")

# Define image size (same as during training)
image_size = (224, 224)

# Define the labels (same order as training)
class_names = ["arrhythmia", "chd", "mi", "normal"]  # Replace with your actual class names

# Load ResNet50 for feature extraction
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Ensure ResNet is frozen

# Function to preprocess and extract features from a single image
def preprocess_image(image_path):
    try:
        # Load image
        img = load_img(image_path, target_size=image_size)
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Extract features using ResNet50
        features = base_model.predict(img_array, verbose=0)
        features = features.reshape(1, -1, features.shape[-1])  # Reshape for LSTM
        
        return features
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Function to make a prediction
def predict_disease(image_path):
    features = preprocess_image(image_path)
    
    if features is not None:
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100  # Confidence percentage
        print(f"Predicted Disease: {class_names[predicted_class]} with {confidence:.2f}% confidence.")
    else:
        print("Prediction failed. Check the input image.")

# Example: Predict from a new image
image_path = "C:\\Users\\TEST\\Desktop\\Project\\ECG\\test\\mi\\MI(233).jpg"  # Change this to your input image path
predict_disease(image_path)
