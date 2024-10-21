# Real-time Audio Emotion Recognition

This project implements a real-time audio emotion recognition system using WebSocket communication between a Python Flask server and a web-based client.

## Project Structure

- `app.py`: Flask server that handles WebSocket connections, processes audio data, and returns emotion predictions.
- `audio_processing.py`: Contains the logic for processing audio data and predicting emotions.
- `templates/index.html`: The main HTML file for the web interface.
- `static/script.js`: Client-side JavaScript for handling audio recording and WebSocket communication.
- `static/styles.css`: CSS file for styling the web interface.

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

   On Linux, you may need to install the following packages before running the above command:
   ```
   sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
   ```

## Usage

1. Dockerfile build:
   ```
   docker build -t pi .   
   ```
2. Run the docker image
   ```
   docker run -p 5001:5001 pi
   ```
2. Open a web browser and navigate to `http://localhost:5001` (or the URL displayed in the console).

3. Click the "Start Recording" button and speak into your microphone.

4. The web interface will display the predicted emotion and confidence level in real-time.

5. Click the "Stop Recording" button to end the session.

## Requirements

- Python 3.7+
- Flask
- Flask-SocketIO
- NumPy
- PyTorch
- Transformers

See `requirements.txt` for a full list of dependencies.

## Note

This project uses the browser's built-in audio capabilities and WebSocket for real-time communication. Ensure your browser supports these features and that you grant the necessary permissions for microphone access when prompted.