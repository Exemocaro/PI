import eventlet
import eventlet.wsgi
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
from audio_processing import process_audio_chunk, SAMPLE_RATE
from datetime import datetime
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, async_mode="eventlet")

BUFFER_DURATION = 5  # Seconds
HALF_BUFFER = BUFFER_DURATION // 2
BUFFER_SIZE = SAMPLE_RATE * BUFFER_DURATION
HALF_BUFFER_SIZE = SAMPLE_RATE * HALF_BUFFER

audio_buffer = np.array([], dtype=np.float32)
overlap_buffer = np.array([], dtype=np.float32)

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
    global audio_buffer, overlap_buffer
    
    audio_chunk = np.frombuffer(data, dtype=np.float32)
    audio_buffer = np.concatenate((audio_buffer, audio_chunk))

    if len(audio_buffer) >= BUFFER_SIZE:
        buffer_to_process = np.concatenate((overlap_buffer, audio_buffer[:BUFFER_SIZE]))

        #Salvar os 2.5s finais para proxima iteração
        overlap_buffer = audio_buffer[HALF_BUFFER_SIZE:BUFFER_SIZE]

        audio_buffer = audio_buffer[HALF_BUFFER_SIZE:] #Remover os primeiros 2.5s

        #Processar o Buffer
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        inicial_time = time.time()
        result = process_audio_chunk(buffer_to_process, timestamp)
        end_time = time.time()

        exec_time = round((end_time - inicial_time), 3)
        print(f"Network result: {result}. Executado em: {exec_time}s")
        
        #Emissão dos resultados
        emit('emotion_result', result)

if __name__ == '__main__':
    print("Starting Flask-SocketIO server")
    socketio.run(app, host="0.0.0.0", port=5001)