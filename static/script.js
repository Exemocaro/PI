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

// FOR THE EMOTION CHART
let emotionChart;
const maxDataPoints = 20;
const emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad'];
const emotionColors = {
    'Anger': '#FF4444',    // Red
    'Disgust': '#9C27B0',  // Purple
    'Fear': '#FFA726',     // Orange
    'Happy': '#4CAF50',    // Green
    'Neutral': '#2196F3',  // Blue
    'Sad': '#607D8B'       // Grey-Blue
};

const emotionData = {
    labels: [],
    datasets: emotions.map(emotion => ({
        label: emotion,
        data: [],
        borderColor: emotionColors[emotion],
        tension: 0.4,
        fill: false,
        borderWidth: 2
    }))
};

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
    socket = io({ transports: ['websocket'] });
    
    socket.on('connect', () => {
        console.log('Connected to server');
    });
    
    
    // Update the socket.on('emotion_result') handler in your startRecording function:
    socket.on('emotion_result', (data) => {
        console.log('Received emotion result:', data);
        let parsedData;
        try {
            parsedData = typeof data === 'string' ? JSON.parse(data) : data;
        } catch (error) {
            console.error('Error parsing emotion result:', error);
            return;
        }
        
        if (parsedData && parsedData.predicted_emotion && parsedData.emotions) {
            // Update the main emotion display
            const mainEmotion = parsedData.predicted_emotion;
            const mainConfidence = parsedData.emotions[mainEmotion];
            
            emotionSpan.textContent = mainEmotion;
            confidenceSpan.textContent = mainConfidence.toFixed(2) + '%';
            
            // Change the emotion text color to match the line color
            emotionSpan.style.color = emotionColors[mainEmotion];

            const timestamp = new Date(parsedData.timestamp);
            const formattedTime = timestamp.toTimeString().split(' ')[0];
                        
            // Update chart data
            emotionData.labels.push(formattedTime);
            
            // Update each emotion's dataset
            emotions.forEach((emotion, index) => {
                emotionData.datasets[index].data.push(parsedData.emotions[emotion]);
            });
            
            // Keep only the last maxDataPoints data points
            if (emotionData.labels.length > maxDataPoints) {
                emotionData.labels.shift();
                emotionData.datasets.forEach(dataset => dataset.data.shift());
            }
            
            // Update chart
            emotionChart.update();
        } else {
            console.error('Invalid emotion result data:', parsedData);
        }
    });

    try {
        console.log("Requesting microphone access...");
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        console.log("Microphone access granted");
        
        // Create an AudioContext with a sample rate of 16,000 Hz
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
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

    // Clear chart data
    emotionData.labels = [];
    emotionData.datasets.forEach(dataset => dataset.data = []);
    emotionChart.update();
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

// EMOTION CHART ------------------------------------------------------------

// Update the initChart function
function initChart() {
    const ctx = document.getElementById('emotionChart').getContext('2d');
    emotionChart = new Chart(ctx, {
        type: 'line',
        data: emotionData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Confidence (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                }
            },
            interaction: {
                mode: 'index',
                intersect: false
            }
        }
    });
}

// Call initChart when the page loads
document.addEventListener('DOMContentLoaded', initChart);
console.log("script.js loaded");