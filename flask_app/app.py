import os
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# --- Model Loading ---
# Define the path to the model relative to the app.py file
MODEL_PATH = '../best_waste_classifier.h5'

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully! ✅")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the class names in the correct order
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


# --- Image Preprocessing ---
def preprocess_image(image_file):
    """Preprocesses the uploaded image to match the model's input requirements."""
    img = Image.open(image_file.stream).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    return img_array / 255.0  # Rescale pixel values


# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction."""
    if model is None:
        return jsonify({'error': 'Model is not loaded.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Preprocess the image
            processed_image = preprocess_image(file)

            # Make prediction
            prediction = model.predict(processed_image)

            # Get the predicted class and confidence
            predicted_class_index = np.argmax(prediction[0])
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = float(np.max(prediction[0])) * 100  # as a percentage

            return jsonify({
                'prediction': predicted_class_name.capitalize(),
                'confidence': f"{confidence:.2f}%"
            })
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500

    return jsonify({'error': 'An unknown error occurred'}), 500


# --- Main ---
if __name__ == '__main__':
    app.run(debug=True)