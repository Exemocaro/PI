const recordButton = document.getElementById('recordButton');
const volumeSpan = document.getElementById('volume');
const emotionSpan = document.getElementById('emotion');
const confidenceSpan = document.getElementById('confidence');
const circleText = document.getElementById('circleText');
const waveContainer = document.querySelector('.wave-container');

const CIRCLE_DIAMETER = 300; // Diameter of the circle in pixels
const WAVE_WIDTH = 2; // Width of each wave bar in pixels
const NUM_WAVES = Math.floor(CIRCLE_DIAMETER / WAVE_WIDTH);
let waves = [];

let isRecording = false;
let socket;
let animationId;
let textHidden = false;

// Create waves
for (let i = 0; i < NUM_WAVES; i++) {
    const wave = document.createElement('div');
    wave.classList.add('wave');
    waveContainer.appendChild(wave);
    waves.push(wave);
}

recordButton.addEventListener('click', () => {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
});

function startRecording() {
    isRecording = true;
    recordButton.textContent = 'Stop Recording';
    recordButton.style.backgroundColor = '#ff4444';
    
    if (!textHidden) {
        circleText.style.opacity = '0';
        textHidden = true;
    }
    
    animateWaves();

    // Connect to WebSocket
    socket = new WebSocket('ws://localhost:8000/ws');
    
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        volumeSpan.textContent = data.volume.toFixed(2);
        emotionSpan.textContent = data.emotion;
        confidenceSpan.textContent = data.confidence.toFixed(2);
    };

    // Start audio recording and sending data
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            const audioContext = new AudioContext();
            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(1024, 1, 1);

            source.connect(processor);
            processor.connect(audioContext.destination);

            processor.onaudioprocess = (e) => {
                if (socket.readyState === WebSocket.OPEN) {
                    const audioData = e.inputBuffer.getChannelData(0);
                    socket.send(audioData);
                }
            };
        })
        .catch(err => console.error('Error accessing microphone:', err));
}

function stopRecording() {
    isRecording = false;
    recordButton.textContent = 'Start Recording';
    recordButton.style.backgroundColor = '#8B7FD3';
    cancelAnimationFrame(animationId);
    waves.forEach(wave => wave.style.height = '10%');

    if (socket) {
        socket.close();
    }
}

function animateWaves() {
    waves.forEach((wave, index) => {
        const distanceFromCenter = Math.abs(index - NUM_WAVES / 2) / (NUM_WAVES / 2);
        const maxHeight = 100 - (distanceFromCenter * 90); // Adjust the multiplier (90) to change the curve
        const height = Math.random() * maxHeight;
        wave.style.height = `${height}%`;
    });
    animationId = requestAnimationFrame(animateWaves);
}