import sounddevice as sd
import numpy as np
import asyncio
import websockets

SAMPLE_RATE = 16000  # Taxa de amostragem do microfone
CHUNK_DURATION = 1   # Duração de cada chunk em segundos
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION  # Tamanho do chunk em amostras

async def send_and_receive_audio(uri):
    async with websockets.connect(uri) as websocket:
        print("Conectado ao servidor WebSocket.")
        
        # Verifica se a conexão foi estabelecida com sucesso
        response = await websocket.recv()
        print("Resposta do servidor:", response)
        
        # Função para capturar o áudio do microfone e enviar ao servidor
        async def send_audio():
            def callback(indata, frames, time, status):
                if status:
                    print(f"Status de áudio: {status}")
                # Converte o áudio em PCM e envia
                pcm_data = indata.astype(np.float32).tobytes()
                asyncio.run_coroutine_threadsafe(websocket.send(pcm_data), loop)

            # Inicia o stream de áudio
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
                print("Capturando e enviando áudio... Pressione Ctrl+C para interromper.")
                await asyncio.Future()  # Mantém o stream ativo até a interrupção

        # Função para receber a resposta do servidor
        async def receive_response():
            while True:
                try:
                    server_response = await websocket.recv()
                    print("Resposta do servidor:", server_response)
                except websockets.ConnectionClosed:
                    print("Conexão com o servidor fechada.")
                    break

        # Inicia as tasks de envio de áudio e recebimento de respostas em paralelo
        await asyncio.gather(send_audio(), receive_response())

# Endereço WebSocket do servidor
uri = "ws://localhost:8000/audio_stream"

# Inicia o loop de eventos e executa o envio e recebimento de áudio
loop = asyncio.get_event_loop()
loop.run_until_complete(send_and_receive_audio(uri))
