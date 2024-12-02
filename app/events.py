import numpy as np
from datetime import datetime
from app import socketio
from app.settings import HALF_BUFFER_SIZE, BUFFER_SIZE, SILENCE_THRESHOLD
from app.utils import calculate_rms
from audio_processing import process_audio_chunk
from flask_socketio import emit
import time
import json

audio_buffer = np.array([], dtype=np.float32)
overlap_buffer = np.array([], dtype=np.float32)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    global audio_buffer, overlap_buffer
    
    audio_buffer = np.array([], dtype=np.float32)
    overlap_buffer = np.array([], dtype=np.float32)   
    print('Client disconnected')

@socketio.on('audio_data')
def handle_audio_data(data):
    global audio_buffer, overlap_buffer
    
    audio_chunk = np.frombuffer(data, dtype=np.float32)
    audio_buffer = np.concatenate((audio_buffer, audio_chunk))

    if len(audio_buffer) >= BUFFER_SIZE:
        buffer_to_process = audio_buffer[:BUFFER_SIZE]

        audio_buffer = audio_buffer[HALF_BUFFER_SIZE:] #Remover os primeiros 2s para um Overlap implicito

        #Detetar Silence
        rms =calculate_rms(buffer_to_process)

        timestamp_slc = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        if rms < SILENCE_THRESHOLD:
            response = {
                "timestamp": timestamp_slc,
                "predicted_emotion": "Silence",
                "emotions": {'Anger': 0, 'Disgust': 0, 'Fear': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Silence': 100}
            }
            emit('emotion_result', json.dumps(response))
        else:
            #Processar o Buffer
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            inicial_time = time.time()
            result = process_audio_chunk(buffer_to_process, timestamp)
            end_time = time.time()

            exec_time = round((end_time - inicial_time), 3)
            print(f"Network result: {result}. Executado em: {exec_time}s")
            
            #Emissão dos resultados
            emit('emotion_result', result)