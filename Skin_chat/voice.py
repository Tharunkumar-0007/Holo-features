from flask import Blueprint, jsonify
import vosk
import sounddevice as sd
import json
import queue
from flask_cors import CORS

voice_bp = Blueprint('voice', __name__)  # Create a Blueprint

# Initialize the Vosk model
model = vosk.Model("./model/vosk-model-small-en-us-0.15")
recognizer = vosk.KaldiRecognizer(model, 16000)

# Queue to hold audio data
audio_queue = queue.Queue()

# Callback function to process audio input
def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    audio_queue.put(bytes(indata))

@voice_bp.route("/start", methods=["GET"])
def start_recognition():
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                           channels=1, callback=callback):
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                recognized_text = result.get("text", "No speech recognized")
                return jsonify({"text": recognized_text})
