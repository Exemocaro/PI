import numpy as np

def calculate_rms(audio_chunk):
    return np.sqrt(np.mean(audio_chunk**2))