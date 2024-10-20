from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
from audio_processing import process_audio_chunk, SAMPLE_RATE
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

BUFFER_DURATION = 5  # Seconds
BUFFER_SIZE = SAMPLE_RATE * BUFFER_DURATION
audio_buffer = np.array([], dtype=np.float32)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('audio_data')
def handle_audio_data(data):
    global audio_buffer
    
    audio_chunk = np.frombuffer(data, dtype=np.float32)
    audio_buffer = np.concatenate((audio_buffer, audio_chunk))

    if len(audio_buffer) >= BUFFER_SIZE:
        buffer_to_process = audio_buffer[:BUFFER_SIZE]
        audio_buffer = audio_buffer[BUFFER_SIZE:]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = process_audio_chunk(buffer_to_process, timestamp)
        print(f"Emotion result: {result}")
        emit('emotion_result', result)

if __name__ == '__main__':
    print("Starting Flask-SocketIO server")
    socketio.run(app, debug=True)