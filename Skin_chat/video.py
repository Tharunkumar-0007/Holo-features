from flask import Blueprint, request, render_template, jsonify
from googleapiclient.discovery import build
from transformers import pipeline

video_bp = Blueprint("video", __name__)

# YouTube Data API Key (replace with your own)
YOUTUBE_API_KEY = 'AIzaSyC8YRIfYGriWYo6DUoRyACvfiHyRgmUsIo'

# Load the Hugging Face pipeline for zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# # Default video links
# default_videos = [
#     "https://www.youtube.com/watch?v=kxc22Fjd1NQ",
#     "https://www.youtube.com/watch?v=hDkY3pZsPGo",
#     "https://www.youtube.com/watch?v=86oF359FHUY",
#     "https://www.youtube.com/watch?v=lHVE4MbfZ6I",
#     "https://www.youtube.com/watch?v=dmbCWXHC3lM",
#     "https://www.youtube.com/watch?v=5V3GPFUzCCk",
# ]

def is_medical_query(query):
    """Check if the query is related to medical topics."""
    candidate_labels = ["Medical", "Non-Medical"]
    result = classifier(query, candidate_labels)
    return result['labels'][0] == 'Medical'

def youtube_search(query):
    """Fetch relevant YouTube videos based on a medical query."""
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(q=query, part='snippet', type='video', maxResults=9)
    response = request.execute()
    
    return [f"https://www.youtube.com/watch?v={item['id']['videoId']}" for item in response['items']]

@video_bp.route('/video_search', methods=['POST'])
def search_videos():
    query = request.form.get('query', '').strip()
    if not query:
        return jsonify({'error': 'Empty query'}), 400

    if is_medical_query(query):
        videos = youtube_search(query)
        return jsonify({'videos': videos})
    else:
        return jsonify({'error': 'Not a medical-related query'}), 400

# @video_bp.route('/videos', methods=['GET'])
# def show_videos():
#     return render_template('videos.html', videos=default_videos)
