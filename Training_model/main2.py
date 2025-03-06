import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from PIL import Image

# Set Paths
train_dir = "C:\\Users\\TEST\\Desktop\\Project\\ECG\\train"
test_dir = "C:\\Users\\TEST\\Desktop\\Project\\ECG\\test"
image_size = (224, 224)  # Standard ResNet input size
num_classes = 4  # Assuming 4 disease categories

# Step 1: Check for Corrupt Images
def check_corrupt_images(directory):
    for class_folder in os.listdir(directory):
        class_path = os.path.join(directory, class_folder)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as e:
                    print(f"Corrupt Image Found: {img_path} - {e}")
                    os.remove(img_path)  # Remove corrupt files

# Run corruption check
check_corrupt_images(train_dir)
check_corrupt_images(test_dir)

# Step 2: Load ECG Images
def load_ecg_images(data_dir):
    images, labels = [], []
    class_names = sorted(os.listdir(data_dir))

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = load_img(img_path, target_size=image_size)
                img_array = img_to_array(img) / 255.0  # Normalize
                images.append(img_array)
                labels.append(class_name)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    return np.array(images), np.array(labels), class_names

# Load Data
X_train, y_train, class_names = load_ecg_images(train_dir)
X_test, y_test, _ = load_ecg_images(test_dir)

# Step 3: Encode Labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Step 4: Feature Extraction with ResNet50
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze pre-trained layers

def extract_features(model, images):
    features = model.predict(images, verbose=1)  # Add verbose for progress display
    return features.reshape(features.shape[0], -1, features.shape[-1])

X_train_features = extract_features(base_model, X_train)
X_test_features = extract_features(base_model, X_test)

# Step 5: Define LSTM Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=X_train_features.shape[1:]),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Step 6: Train Model
model.fit(X_train_features, y_train, validation_data=(X_test_features, y_test), epochs=10, batch_size=16)

# Step 7: Evaluate Model
loss, accuracy = model.evaluate(X_test_features, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 8: Save Model
model.save("ecg_lstm_model.h5")

print("Model Training Complete and Saved as 'ecg_lstm_model.h5' ✅")

