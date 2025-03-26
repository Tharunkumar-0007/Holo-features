from flask import Blueprint, request, render_template, jsonify
from googleapiclient.discovery import build
from transformers import pipeline

video_bp = Blueprint("video", __name__)

# YouTube Data API Key (replace with your own)
YOUTUBE_API_KEY = 'AIzaSyAsaqtSVvfAa8iXCdUf2S7jHS837IK5ibQ'

# Load the Hugging Face pipeline for zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

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
    
    videos = []
    for item in response['items']:
        video_data = {
            "title": item['snippet']['title'],
            "channel": item['snippet']['channelTitle'],
            "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        }
        videos.append(video_data)

    return videos

@video_bp.route('/video_search', methods=['POST'])
def search_videos():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Empty or invalid JSON request'}), 400

    query = data['query'].strip()
    if not query:
        return jsonify({'error': 'Empty query'}), 400

    if is_medical_query(query):
        videos = youtube_search(query)
        return jsonify({'videos': videos})
    else:
        return jsonify({'error': 'Not a medical-related query'}), 400