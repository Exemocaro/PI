import asyncio
import websockets
import pyaudio
import json

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

async def record_and_send():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("* Recording. Speak into your microphone. Press Ctrl+C to stop.")
        
        try:
            while True:
                data = stream.read(CHUNK)
                await websocket.send(data)
                
                response = await websocket.recv()
                result = json.loads(response)
                volume = result['volume']
                emotion = result['emotion']
                confidence = result['confidence']
                
                print(f"Volume: {volume:.2f} Predicted emotion: {emotion}, confidence: {confidence:.2f}")
        except KeyboardInterrupt:
            print("* Stopped recording")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(record_and_send())