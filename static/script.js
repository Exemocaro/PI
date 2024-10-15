const recordButton = document.getElementById('recordButton');
const emotionSpan = document.getElementById('emotion');
const confidenceSpan = document.getElementById('confidence');
const circleText = document.getElementById('circleText');
const waveContainer = document.querySelector('.wave-container');

const CIRCLE_DIAMETER = 300;
const WAVE_WIDTH = 2;
const NUM_WAVES = Math.floor(CIRCLE_DIAMETER / WAVE_WIDTH);
let waves = [];

let isRecording = false;
let socket;
let animationId;
let textHidden = false;
let audioContext;
let source;
let processor;

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

async function startRecording() {
    console.log("Starting recording...");
    isRecording = true;
    recordButton.textContent = 'Stop Recording';
    recordButton.style.backgroundColor = '#ff4444';
    
    if (!textHidden) {
        circleText.style.opacity = '0';
        textHidden = true;
    }
    
    animateWaves();

    // Connect to SocketIO
    socket = io();
    
    socket.on('connect', () => {
        console.log('Connected to server');
    });
    
    socket.on('emotion_result', (data) => {
        console.log('Received emotion result:', data);
        let parsedData;
        try {
            parsedData = typeof data === 'string' ? JSON.parse(data) : data;
        } catch (error) {
            console.error('Error parsing emotion result:', error);
            return;
        }
        
        if (parsedData && parsedData.emotion && parsedData.confidence !== undefined) {
            emotionSpan.textContent = parsedData.emotion;
            confidenceSpan.textContent = parsedData.confidence.toFixed(2) + '%';
        } else {
            console.error('Invalid emotion result data:', parsedData);
        }
    });

    try {
        console.log("Requesting microphone access...");
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        console.log("Microphone access granted");
        
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        source = audioContext.createMediaStreamSource(stream);
        processor = audioContext.createScriptProcessor(1024, 1, 1);

        source.connect(processor);
        processor.connect(audioContext.destination);

        processor.onaudioprocess = (e) => {
            if (isRecording && socket.connected) {
                const audioData = e.inputBuffer.getChannelData(0);
                socket.emit('audio_data', audioData);
            }
        };
    } catch (err) {
        console.error('Error accessing microphone:', err);
    }
}

function stopRecording() {
    console.log("Stopping recording...");
    isRecording = false;
    recordButton.textContent = 'Start Recording';
    recordButton.style.backgroundColor = '#8B7FD3';
    cancelAnimationFrame(animationId);
    waves.forEach(wave => wave.style.height = '10%');

    if (socket) {
        socket.disconnect();
    }

    if (source) {
        source.disconnect();
    }
    if (processor) {
        processor.disconnect();
    }
    if (audioContext) {
        audioContext.close();
    }
}

function animateWaves() {
    waves.forEach((wave, index) => {
        const distanceFromCenter = Math.abs(index - NUM_WAVES / 2) / (NUM_WAVES / 2);
        const maxHeight = 100 - (distanceFromCenter * 90);
        const height = Math.random() * maxHeight;
        wave.style.height = `${height}%`;
    });
    animationId = requestAnimationFrame(animateWaves);
}

console.log("script.js loaded");