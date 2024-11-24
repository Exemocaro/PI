import numpy as np
import json
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import os
from app.settings import SAMPLE_RATE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o feature_extractor e o modelo pré-treinado
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er").to(DEVICE)
model.eval()  # Colocar o modelo em modo de avaliação

# Obter o mapeamento das labels do modelo (abreviações)
id2label_abbr = model.config.id2label

# Criar um mapeamento de abreviações para nomes completos das emoções
abbr_to_full = {
    'ang': 'Angry',
    'dis': 'Disgust',
    'fea': 'Fear',
    'hap': 'Happy',
    'neu': 'Neutral',
    'sad': 'Sad',
}

# Criar o mapeamento final de id para nomes completos
id2label = {int(k): abbr_to_full[v] for k, v in id2label_abbr.items()}

def process_audio_chunk(pcm_data: np.ndarray, timestamp: float) -> str:
    """Processa um chunk de áudio e realiza o reconhecimento de emoções.

    Retorna os valores de confiança para todas as emoções juntamente com a emoção predita.
    """
    # Garantir que pcm_data seja um array numpy unidimensional
    pcm_data = np.squeeze(pcm_data)

    # Converter pcm_data para float32 e normalizar se necessário
    if pcm_data.dtype != np.float32:
        pcm_data = pcm_data.astype(np.float32) / np.iinfo(np.int16).max

    # Processar os dados de áudio com o feature_extractor
    inputs = feature_extractor(
        [pcm_data],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    # Enviar inputs para o dispositivo adequado
    inputs = {key: inputs[key].to(DEVICE) for key in inputs}

    # Realizar inferência com o modelo
    with torch.no_grad():
        logits = model(**inputs).logits

    # Aplicar softmax para obter as probabilidades de todas as emoções
    probabilities = torch.softmax(logits, dim=-1)[0]

    # Obter a emoção predita (maior probabilidade)
    predict_id = torch.argmax(probabilities).item()
    predicted_emotion = id2label[predict_id]

    # Criar um dicionário de todas as emoções e suas porcentagens de confiança
    emotion_confidences = {
        id2label[i]: round(probabilities[i].item() * 100, 2)
        for i in range(len(probabilities))
    }

    # Retornar o resultado como uma string JSON com o timestamp
    response = {
        "timestamp": timestamp,
        "predicted_emotion": predicted_emotion,
        "emotions": emotion_confidences
    }

    return json.dumps(response)
