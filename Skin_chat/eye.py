from flask import Blueprint, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import io

# Create a Blueprint for eye disease prediction
eye_bp = Blueprint('eye_bp', __name__, template_folder='templates')

# Load the model
model = load_model('model/eye_disease_model.h5')

class_names = ['Cataract', 'Diabetic_Retinopathy', 'Glaucoma', 'Normal']
confidence_threshold = 0.7  # Set a threshold for valid predictions

def is_medical_image(file):
    """
    Rule-based function to check if the image is likely a medical eye image.
    """
    file.seek(0)  # Reset file pointer before reading
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return False  # Invalid image file

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate image statistics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    # Rule-based thresholding
    if mean_intensity < 50 or mean_intensity > 200:
        return False
    if std_intensity < 10 or std_intensity > 100:
        return False

    return True  # Likely an eye image

def predict_image(file):
    """
    Predict the eye disease if the image is valid.
    """
    file.seek(0)  # Reset file pointer before reading
    img = image.load_img(io.BytesIO(file.read()), target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    confidence = np.max(prediction)
    predicted_class = class_names[np.argmax(prediction)]

    if confidence < confidence_threshold:
        return "Not a valid eye image", confidence
    return predicted_class, confidence

@eye_bp.route('/eye', methods=['POST'])
def index():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file:
        if not is_medical_image(file):
            return jsonify({'message': 'Give a valid eye image'}), 400

        prediction, confidence = predict_image(file)
        confidence_percentage = round(confidence * 100, 2)

        return jsonify({'prediction': prediction, 'confidence': confidence_percentage})

    return jsonify({'message': 'File processing error'}), 500

