def predict_from_volume(volume):
    if volume < 100:
        return "silent", 0.9
    elif volume < 500:
        return "calm", 0.8
    elif volume < 1000:
        return "neutral", 0.7
    elif volume < 2000:
        return "animated", 0.8
    elif volume < 5000:
        return "excited", 0.9
    else:
        return "shouting", 1.0
    