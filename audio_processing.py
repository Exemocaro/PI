import numpy as np
import wave
import io
import json
import torch
import torch.nn as nn
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import gdown
import os

SAMPLE_RATE = 16000
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

# Initialize the feature extractor with wavlm-base
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")

# Initialize the WavLM model with wavlm-base
wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base").to(DEVICE)

# Define the custom model class
class WavLMClassifierWithRNN(nn.Module):
    def __init__(self, wavlm_model, hidden_size=256, rnn_type='GRU'):
        super(WavLMClassifierWithRNN, self).__init__()
        self.wavlm = wavlm_model
        self.hidden_size = hidden_size

        # Define the RNN layer
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=wavlm_model.config.hidden_size,
                hidden_size=hidden_size,
                batch_first=True
            )
        else:
            self.rnn = nn.LSTM(
                input_size=wavlm_model.config.hidden_size,
                hidden_size=hidden_size,
                batch_first=True
            )

        # Define the classification layer
        self.classifier = nn.Linear(hidden_size, 6)  # Assuming 6 emotion classes

    def forward(self, input_values):
        # Get the hidden states from WavLM
        with torch.no_grad():
            outputs = self.wavlm(input_values)
            hidden_states = outputs.last_hidden_state

        # Pass through the RNN
        rnn_output, _ = self.rnn(hidden_states)

        # Aggregate the RNN outputs (e.g., by averaging)
        rnn_output = rnn_output.mean(dim=1)

        # Get the logits from the classifier
        logits = self.classifier(rnn_output)
        return logits

file_id = '1cc7V6thKP6-fxQN2nhMLq2j1qG5COgtt'
output_path = 'wavlm_rnn_model.pt'
if not os.path.exists(output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)


# Instantiate the model and load the trained weights
model = WavLMClassifierWithRNN(wavlm_model).to(DEVICE)
checkpoint = torch.load(output_path, map_location=DEVICE)
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
    inputs = feature_extractor(
        pcm_data,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )
    input_values = inputs['input_values'].to(DEVICE)

    # Perform inference with the model
    with torch.no_grad():
        logits = model(input_values)

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
