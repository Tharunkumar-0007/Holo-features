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

# Initialize Flask app and configurations
app = Flask(__name__)
CORS(app)

# Register the blueprints
app.register_blueprint(mental_age_bp)
app.register_blueprint(eye_bp)
app.register_blueprint(book_bp) 
app.register_blueprint(video_bp)  

df = pd.read_csv("model/drugs.csv")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()  # Get JSON data from the request
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400

    user_input = data['query'].strip()
    if not user_input or not re.search(r'[a-zA-Z0-9]', user_input):
        return jsonify({"response": "Nothing matched. Please enter a valid query."})

    response = ask_question(user_input)
    return jsonify({"response": response})


ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.jfif'}
def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        app.logger.error('No file part in request')
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        app.logger.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Call the prediction function from skin.py
        result = predict_skin_disease(file)

        # If the result contains a message (like invalid image), return that
        if 'message' in result:
            return jsonify({'error': result['message']}), 400

        app.logger.info(f"Prediction result: {result}")
        return jsonify(result), 200

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route("/search", methods=["POST"])
def search():
    # Get the drug name entered by the user
    drug_name = request.form.get("drug_name").strip().lower()
    result = df[df['drugName'].str.lower() == drug_name]

    if not result.empty:
        response = {
            "url": result.iloc[0]["url"],
            "description": result.iloc[0]["description"]
        }
        return jsonify(response)
    else:
        # If no match is found, return an error response
        return jsonify({"error": "Drug not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000, use_reloader=False)
    # app.run(host='0.0.0.0', debug=True, port=5000)
