import os
import json
import logging
from pathlib import Path
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np

# --- Initialization ---
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# --- Configuration ---
# Use Path for robust path handling
APP_ROOT = Path(__file__).parent
MODEL_PATH = APP_ROOT / '../best_waste_classifier.h5'
CLASSES_PATH = APP_ROOT / '../class_indices.json'

# --- Model and Class Loading ---
def load_model_and_classes():
    """Loads the trained model and class names."""
    model = None
    class_names = None
    
    # Load model
    if MODEL_PATH.exists():
        try:
            model = tf.keras.models.load_model(str(MODEL_PATH))
            app.logger.info("Model loaded successfully! ✅")
        except Exception as e:
            app.logger.error(f"Error loading model: {e}", exc_info=True)
    else:
        app.logger.error(f"Model file not found at {MODEL_PATH}")

    # Load class names from JSON
    if CLASSES_PATH.exists():
        try:
            with open(CLASSES_PATH, 'r') as f:
                # Sort by value to ensure order is correct
                class_indices = json.load(f)
                class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
            app.logger.info(f"Class names loaded: {class_names}")
        except Exception as e:
            app.logger.error(f"Error loading class names from {CLASSES_PATH}: {e}", exc_info=True)
    else:
        app.logger.error(f"Class indices file not found at {CLASSES_PATH}. Using default.")
        # Fallback to hardcoded list if file not found
        class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    return model, class_names

model, CLASS_NAMES = load_model_and_classes()

# --- Image Preprocessing ---
def preprocess_image(image_stream):
    """Preprocesses the uploaded image to match the model's input requirements."""
    try:
        img = Image.open(image_stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.asarray(img)
        img_array = np.expand_dims(img_array, axis=0)  # Create a batch
        return img_array / 255.0  # Rescale
    except Exception as e:
        app.logger.error(f"Error preprocessing image: {e}", exc_info=True)
        return None

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction."""
    if model is None:
        return jsonify({'error': 'Model is not available or failed to load.'}), 503 # Service Unavailable
    if CLASS_NAMES is None:
        return jsonify({'error': 'Class names are not available.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if file:
        processed_image = preprocess_image(file.stream)
        if processed_image is None:
            return jsonify({'error': 'Failed to process image.'}), 400

        try:
            # Make prediction
            prediction = model.predict(processed_image)

            # Get predicted class and confidence
            predicted_class_index = np.argmax(prediction[0])
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = float(np.max(prediction[0])) * 100

            return jsonify({
                'prediction': predicted_class_name.capitalize(),
                'confidence': f"{confidence:.2f}%"
            })
        except Exception as e:
            app.logger.error(f"Prediction failed: {e}", exc_info=True)
            return jsonify({'error': 'An error occurred during prediction.'}), 500

    return jsonify({'error': 'An unknown error occurred.'}), 500


if __name__ == '__main__':
    # Use waitress or gunicorn in production instead of app.run()
    app.run(debug=True, host='0.0.0.0', port=5000)
