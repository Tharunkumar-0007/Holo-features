from flask import Flask, render_template, request, jsonify
import re
from flask_cors import CORS
import pandas as pd
from skin import predict_skin_disease
from chat import ask_question
from mental_age import mental_age_bp  
from eye import eye_bp  
from book import book_bp  
from video import video_bp
from voice import voice_bp
import logging
from werkzeug.utils import secure_filename
import os

# Initialize Flask app and configurations
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register blueprints
app.register_blueprint(mental_age_bp)
app.register_blueprint(eye_bp)
app.register_blueprint(book_bp)
app.register_blueprint(video_bp)
app.register_blueprint(voice_bp)

# Load drugs data safely
try:
    df = pd.read_csv("model/drugs.csv")
    df.dropna(subset=['drugName'], inplace=True)  # Handle missing values
    logger.info("Successfully loaded drugs data")
except Exception as e:
    logger.error(f"Failed to load drugs data: {str(e)}")
    df = pd.DataFrame(columns=['drugName', 'url', 'description'])

# File upload settings
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.jfif'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400

        user_input = data['query'].strip()
        if not user_input or not re.search(r'[a-zA-Z0-9]', user_input):
            return jsonify({"response": "Please enter a valid query."})

        logger.info(f"Processing question: {user_input}")
        response = ask_question(user_input)

        if not response:
            return jsonify({"error": "No response available"}), 500

        return jsonify({"response": response})

    except Exception as e:
        logger.error(f"Error in /ask endpoint: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            logger.error('No file uploaded')
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            logger.error('No file selected')
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            logger.error('Invalid file type')
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        logger.info(f"File saved at: {filepath}")

        # Call the prediction function
        result = predict_skin_disease(filepath)

        # Remove the file after processing
        try:
            os.remove(filepath)
        except Exception as e:
            logger.warning(f"Could not delete file {filepath}: {str(e)}")

        if 'error' in result:
            logger.error(f"Prediction error: {result['error']}")
            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route("/search", methods=["POST"])
def search():
    try:
        drug_name = request.form.get("drug_name", "").strip().lower()
        if not drug_name:
            return jsonify({"error": "No drug name provided"}), 400

        result = df[df['drugName'].str.lower() == drug_name]

        if not result.empty:
            return jsonify({
                "url": result.iloc[0]["url"],
                "description": result.iloc[0]["description"]
            })

        # Suggest similar drugs
        suggestions = df['drugName'].str.lower().unique()[:5].tolist()
        return jsonify({"error": "Drug not found", "suggestions": suggestions}), 404

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
