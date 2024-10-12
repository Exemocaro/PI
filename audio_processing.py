import numpy as np
import wave
import io
import json
from transformers import Wav2Vec2Processor, HubertForSequenceClassification, AutoFeatureExtractor
import torch

SAMPLE_RATE = 16000
MODEL_NAME = "superb/hubert-base-superb-er"

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = HubertForSequenceClassification.from_pretrained(MODEL_NAME)


def pcm_to_wav(pcm_data: bytes) -> bytes:
    """Converte PCM para formato WAV."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Canal mono
        wav_file.setsampwidth(2)  # 16 bits
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm_data)
    wav_buffer.seek(0)
    return wav_buffer.read()

def process_audio_chunk(pcm_data: np.ndarray, timestamp: float) -> str:
    """Processa um chunk de áudio e analisa com a rede neural."""
    #wav_data = pcm_to_wav(pcm_data)

    #Analise rede neuronal: Input array
    inputs = feature_extractor(pcm_data, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding = True)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predict_id = torch.argmax(logits, dim=-1).item()
    emotion = model.config.id2label[predict_id]
    confidence = torch.softmax(logits, dim=-1)[0][predict_id].item()
    
    #Converte a confiança para percentagem
    confidence_percent = round(confidence * 100, 2)
    
    # Retorne o resultado em formato JSON com o timestamp
    response = {
        "timestamp": timestamp,
        "emotion": emotion,
        "confidence": confidence_percent
    }
    
    return json.dumps(response)
