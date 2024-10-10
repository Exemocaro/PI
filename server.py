from fastapi import FastAPI, WebSocket
import uvicorn
import numpy as np
import asyncio
from datetime import datetime

from audio_processing import process_audio_chunk, SAMPLE_RATE

app = FastAPI()
connected_clients = set()
BUFFER_DURATION = 5 #Seconds
BUFFER_SIZE = SAMPLE_RATE * 5

@app.websocket("/audio_stream")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"status": 200, "message": "connected!"})
    connected_clients.add(websocket)
    print(f"Client connected. Total clients: {len(connected_clients)}")

    audio_buffer = np.array([], dtype=np.float32)


    try:
        while True:
                data = await websocket.receive_bytes()

                audio_chunk = np.frombuffer(data, np.float32)

                audio_buffer = np.concatenate((audio_buffer,audio_chunk))

                if len(audio_buffer) >= BUFFER_SIZE:
                     buffer_to_process = audio_buffer[:BUFFER_SIZE]
                     audio_buffer = audio_buffer[BUFFER_SIZE:]

                     asyncio.create_task(process_and_send_results(buffer_to_process,websocket))

                # timestamp = time.time()
                # response_json = process_audio_chunk(chunk, timestamp)
                # if response_json:
                #     await websocket.send_json(response_json)
    except Exception as e:
        print(f"Client disconnected: {e}")
    finally:
        connected_clients.remove(websocket)
        print(f"Client disconnected. Total clients: {len(connected_clients)}")


async def process_and_send_results(buffer, websocket):
        try:
          timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
          result = process_audio_chunk(buffer,timestamp)

          await websocket.send_json(result)
        except Exception as e:
            print(f"Error {e}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)