# Real-time Audio Emotion Recognition

This project implements a real-time audio emotion recognition system using WebSocket communication between a Python client and server.

## Project Structure

- `client.py`: Records audio from the microphone and sends it to the server for emotion analysis.
- `server.py`: Receives audio data, processes it, and returns emotion predictions.
- `emotion_predictor.py`: Contains the logic for predicting emotions based on audio volume.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment:
   ```
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     .venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source .venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

   In Linux, please install the following packages before running the above command (in case it doesn't work):
   ```
   sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
   ```

## Usage

1. Start the server:
   ```
   python server.py
   ```

2. In a separate terminal, run the client:
   ```
   python client.py
   ```

3. Speak into your microphone. The client will display the predicted emotion and confidence level in real-time.

4. Press Ctrl+C to stop the client.
