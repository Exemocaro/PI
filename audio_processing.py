import numpy as np
import wave
import io
import json
import torch
import torch.nn as nn
from transformers import WhisperProcessor, WhisperModel
import gdown
import os
from app.settings import SAMPLE_RATE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the mapping from class indices to emotion labels
id2label = {
    0: 'Anger',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad'
}


whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
whisper_model = WhisperModel.from_pretrained("openai/whisper-base").to(DEVICE)

# Define the custom model class
class WhisperClassifierWithRNN(nn.Module):
    def __init__(self, whisper_model, hidden_size=256, rnn_type='LSTM'):
        super(WhisperClassifierWithRNN, self).__init__()
        self.whisper = whisper_model
        self.hidden_size = hidden_size

        # Use either LSTM or GRU based on rnn_type parameter
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=whisper_model.config.d_model, hidden_size=hidden_size, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=whisper_model.config.d_model, hidden_size=hidden_size, batch_first=True)

        # Classificador com uma camada linear simples
        self.classifier = nn.Linear(hidden_size, 6)

    def forward(self, input_features):
        with torch.no_grad():
            # Use apenas o encoder do modelo Whisper
            encoder_outputs = self.whisper.encoder(input_features)
            hidden_states = encoder_outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]

        # Passar pelo RNN (LSTM ou GRU)
        rnn_output, _ = self.rnn(hidden_states)  # Shape: [batch_size, seq_len, hidden_size]

        # Agregar tirando a média ao longo do comprimento da sequência
        rnn_output = rnn_output.mean(dim=1)  # Shape: [batch_size, hidden_size]

        # Classificar usando a camada linear
        logits = self.classifier(rnn_output)
        return logits

file_id = '1cKG2sQImKSdPWv9AUZIEslRO97Rg1ozr'
output_path = 'whisper_base3_rnn_model.pt'
if not os.path.exists(output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)


# Instantiate the model and load the trained weights
model = WhisperClassifierWithRNN(whisper_model).to(DEVICE)
checkpoint = torch.load(output_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation mode

#LEGACY CODE - SE NÃO USAR NO FINAL, RETIRAR!!
# def pcm_to_wav(pcm_data: bytes) -> bytes:
#     """Convert PCM data to WAV format."""
#     wav_buffer = io.BytesIO()
#     with wave.open(wav_buffer, 'wb') as wav_file:
#         wav_file.setnchannels(1)  # Mono channel
#         wav_file.setsampwidth(2)  # 16 bits per sample
#         wav_file.setframerate(SAMPLE_RATE)
#         wav_file.writeframes(pcm_data)
#     wav_buffer.seek(0)
#     return wav_buffer.read()

def process_audio_chunk(pcm_data: np.ndarray, timestamp: float) -> str:
    """Process an audio chunk and perform emotion recognition.
    
    Returns confidence values for all emotions along with the predicted emotion.
    """
    # Extract features from the audio chunk
    inputs = whisper_processor(
        pcm_data,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    input_features = inputs['input_features'].to(DEVICE)

    # Pad or truncate input_features to length 3000
    if input_features.shape[-1] < 3000:
        pad_length = 3000 - input_features.shape[-1]
        input_features = torch.nn.functional.pad(
            input_features, 
            (0, pad_length),  # Pad only on the right side (end of the sequence)
            mode='constant', 
            value=0
        )
    elif input_features.shape[-1] > 3000:
        input_features = input_features[:, :, :3000]  # Truncate to 3000

    # Perform inference with the model
    with torch.no_grad():
        logits = model(input_features)

    # Apply softmax to get probabilities for all emotions
    probabilities = torch.softmax(logits, dim=-1)[0]
    
    # Get the predicted emotion (highest probability)
    predict_id = torch.argmax(probabilities).item()
    predicted_emotion = id2label[predict_id]
    
    # Create dictionary of all emotions and their confidence percentages
    emotion_confidences = {
        id2label[i]: round(probabilities[i].item() * 100, 2)
        for i in range(len(id2label))
    }

    emotion_confidences["Silence"] = 0

    # Return the result as a JSON string with the timestamp
    response = {
        "timestamp": timestamp,
        "predicted_emotion": predicted_emotion,
        "emotions": emotion_confidences
    }

    return json.dumps(response)

