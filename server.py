from fastapi import FastAPI, WebSocket
import uvicorn
import numpy as np
import json

from emotion_predictor import *

app = FastAPI()
connected_clients = set()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"Client connected. Total clients: {len(connected_clients)}")

    try:
        while True:
            data = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_chunk).mean()
            emotion, confidence = predict_from_volume(volume)
            
            response = json.dumps({
                "volume": float(volume),
                "emotion": emotion,
                "confidence": confidence
            })
            await websocket.send_text(response)
    except Exception as e:
        print(f"Client disconnected: {e}")
    finally:
        connected_clients.remove(websocket)
        print(f"Client disconnected. Total clients: {len(connected_clients)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)