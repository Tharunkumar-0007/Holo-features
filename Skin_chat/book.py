from flask import Blueprint, jsonify, request
from transformers import pipeline
import requests

# Initialize the Blueprint
book_bp = Blueprint('book_bp', __name__)

# Load the Hugging Face pipeline for zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Google Books API URL (replace with your own)
API_URL = "https://www.googleapis.com/books/v1/volumes?q=subject:{}&key=AIzaSyAsaqtSVvfAa8iXCdUf2S7jHS837IK5ibQ"

def is_medical_query(query):
    # Specify candidate labels for zero-shot classification
    candidate_labels = ["Medical", "Non-Medical"]
    result = classifier(query, candidate_labels)
    return result['labels'][0] == 'Medical'

@book_bp.route('/suggest', methods=['POST'])
def suggest():
    # Get the genre from the request
    data = request.get_json()
    genre = data.get('genre', '').lower()

    # Check if the genre is medical-related using the classifier
    if is_medical_query(genre):
        # Fetch book data from Google Books API
        response = requests.get(API_URL.format(genre))
        data = response.json()

        if 'items' in data:
            # Limit to first 14 books
            suggested_books = [{'title': book['volumeInfo'].get('title', 'No Title'),
                                'authors': book['volumeInfo'].get('authors', ['Unknown Author']),
                                'thumbnail': book['volumeInfo'].get('imageLinks', {}).get('thumbnail', ''),
                                'infoLink': book['volumeInfo'].get('infoLink', '')}
                               for book in data['items'][:14]] 
            return jsonify(suggested_books=suggested_books)
        else:
            return jsonify(suggested_books=[], message="No books found!")
    else:
        return jsonify(suggested_books=[], message="Not medical-related.")
