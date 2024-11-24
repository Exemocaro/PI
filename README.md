# Real-time Audio Emotion Recognition

This project implements a real-time audio emotion recognition system using WebSocket communication between a Python Flask server and a web-based client. The system provides visual feedback and real-time graphing of emotion predictions.

## Project Structure

- `app/...`: Folder containing all the necessary files to run the server.
   - `app/__init__.py`: The inicialization of the API.
   - `app/events.py`: Handle the events of the API like connection and disconection of clients and handles the audio data to send the neural network.
   - `app/routes.py`: The route for the server, renders the inicial and only page of this API.
   - `app/settings.py`: Responsible for the global variables of the API, like Sample Rate, Buffer Duration and Silence Threshold.
   - `app/utils.py`: Handles the others functions necessary for the processing of the audio.
- `run.py`: Start the server.
- `audio_processing.py`: Contains the logic for processing audio data and predicting emotions using WavLM and RNN models.
- `templates/index.html`: The main HTML file for the web interface.
- `static/script.js`: Client-side JavaScript for handling audio recording, WebSocket communication, and chart visualization.
- `static/styles.css`: CSS file for styling the web interface.
- `samples/`: Contains sample audio files for testing the emotion recognition system.

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

### Option 1: Run with Flask

1. Start the Flask server:
   ```bash
   python run.py
   ```
2. Open a web browser and navigate to `http://localhost:5001` (or the URL displayed in the console).

3. Click the "Start Recording" button and speak into your microphone.

4. The web interface will display:
   - Real-time audio visualization
   - Current predicted emotion and confidence level
   - A line chart showing the confidence levels of all emotions over time

5. Click the "Stop Recording" button to end the session.

### Option 2: Run with Docker

1. Build the Docker image:
   ```bash
   docker build -t pi .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 5001:5001 pi
   ```
3. Open a web browser and navigate to `http://localhost:5001` (or the URL displayed in the console).

4. Follow the same steps as above to use the emotion recognition system.

### Option 3: Run with Docker-Compose

1. Build the project with the following command:
   ```bash
   docker compose up
   ```
2. Open a web browser and navigate to ``http://localhost:5001` (or the URL displayed in the console).`

## Requirements

- Python 3.10+
- Flask
- Flask-SocketIO
- NumPy
- PyTorch
- Transformers
- Chart.js (included via CDN)
- Socket.IO client (included via CDN)

See `requirements.txt` for a full list of dependencies.

## Note

This project uses the browser's built-in audio capabilities and WebSocket for real-time communication. Ensure your browser supports these features and that you grant the necessary permissions for microphone access when prompted. The visualization features work best in modern browsers with good WebSocket support. Samples were taken from the [Sample Focus Website](https://samplefocus.com/).
