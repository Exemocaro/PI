const recordButton = document.getElementById('recordButton');
const discardButton = document.getElementById('discardButton');
const uploadButton = document.getElementById('uploadButton');
const pauseButton = document.getElementById('pauseButton');
const emotionSpan = document.getElementById('emotion');
const confidenceSpan = document.getElementById('confidence');
const circleText = document.getElementById('circleText');
const waveContainer = document.querySelector('.wave-container');

const fileInput = document.createElement('input');
fileInput.type = 'file';
fileInput.accept = 'audio/*';
fileInput.style.display = 'none';
document.body.appendChild(fileInput);

recordButton.parentNode.appendChild(uploadButton);

const CIRCLE_DIAMETER = 300;
const WAVE_WIDTH = 2;
const NUM_WAVES = Math.floor(CIRCLE_DIAMETER / WAVE_WIDTH);
let waves = [];

let isRecording = false;
let isPlaying = false;
let isPaused = false;
let socket;
let animationId;
let textHidden = false;
let audioContext;
let source;
let processor;
let audioBuffer;
let startTime;
let currentTime = 0;



const maxDataPoints = 20;
const emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Silence'];
const emotionColors = {
    'Anger': '#FF4444',
    'Disgust': '#9C27B0',
    'Fear': '#FFA726',
    'Happy': '#4CAF50',
    'Neutral': '#2196F3',
    'Sad': '#607D8B',
    'Silence': '#000000'
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

for (let i = 0; i < NUM_WAVES; i++) {
    const wave = document.createElement('div');
    wave.classList.add('wave');
    waveContainer.appendChild(wave);
    waves.push(wave);
}

pauseButton.addEventListener('click', () => {
    if (isPaused) {
        resumePlayback();
    } else {
        pausePlayback();
    }
});

uploadButton.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
        // First create AudioContext with default sample rate
        const tempContext = new (window.AudioContext || window.webkitAudioContext)();
        const arrayBuffer = await file.arrayBuffer();
        const originalAudioBuffer = await tempContext.decodeAudioData(arrayBuffer);
        
        // Create offline context for resampling
        const offlineContext = new OfflineAudioContext({
            numberOfChannels: 1,
            length: Math.ceil(originalAudioBuffer.duration * 16000),
            sampleRate: 16000
        });

        // Create buffer source
        const source = offlineContext.createBufferSource();
        source.buffer = originalAudioBuffer;
        source.connect(offlineContext.destination);
        source.start();

        // Render the resampled audio
        const resampledBuffer = await offlineContext.startRendering();
        
        // Now create the actual AudioContext we'll use for playback
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        audioBuffer = resampledBuffer;
        
        recordButton.textContent = 'Play Audio';
        recordButton.style.backgroundColor = '#4CAF50';
        recordButton.disabled = false;
        
        // Show discard button, hide upload button
        discardButton.style.display = 'inline-block';
        uploadButton.style.display = 'none';
    } catch (error) {
        console.error('Error loading audio file:', error);
        alert('Error loading audio file. Please ensure the file is a valid audio format and try again.');
    }
});

recordButton.addEventListener('click', () => {
    if (audioBuffer) {
        if (!isPlaying) {
            startPlayback();
        } else {
            stopPlayback();
        }
    } else if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
});

discardButton.addEventListener('click', discardAudio);

async function startPlayback() {
    console.log("Starting playback...");
    isPlaying = true;
    isPaused = false;
    recordButton.textContent = 'Stop';
    recordButton.style.backgroundColor = '#ff4444';
    pauseButton.style.display = 'inline-block';
    pauseButton.textContent = 'Pause';
    
    if (!textHidden) {
        circleText.style.opacity = '0';
        textHidden = true;
    }
    
    animateWaves();

    socket = io({ transports: ['websocket'] });
    
    socket.on('connect', () => {
        console.log('Connected to server');
    });
    
    socket.on('emotion_result', handleEmotionResult);

    source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    
    processor = audioContext.createScriptProcessor(1024, 1, 1);
    source.connect(processor);
    processor.connect(audioContext.destination);
    source.connect(audioContext.destination);

    startTime = audioContext.currentTime;
    currentTime = 0;

    processor.onaudioprocess = (e) => {
        if (isPlaying && socket.connected) {
            const audioData = e.inputBuffer.getChannelData(0);
            socket.emit('audio_data', audioData);
        }
    };

    source.onended = () => {
        stopPlayback();
    };

    source.start(0);
}

function pausePlayback() {
    if (isPlaying && !isPaused) {
        console.log("Pausing playback...");
        isPaused = true;
        source.playbackRate.value = 0; // This effectively pauses the audio
        cancelAnimationFrame(animationId);
        pauseButton.textContent = 'Resume';
        waves.forEach(wave => wave.style.height = '10%');
    }
}

function resumePlayback() {
    if (isPlaying && isPaused) {
        console.log("Resuming playback...");
        isPaused = false;
        source.playbackRate.value = 1; // Restore normal playback
        animateWaves();
        pauseButton.textContent = 'Pause';
    }
}

function stopPlayback() {
    console.log("Stopping playback...");
    isPlaying = false;
    isPaused = false;
    recordButton.textContent = 'Play Audio';
    recordButton.style.backgroundColor = '#4CAF50';
    pauseButton.style.display = 'none';
    cancelAnimationFrame(animationId);
    waves.forEach(wave => wave.style.height = '10%');

    if (socket) {
        socket.disconnect();
    }

    if (source) {
        source.stop();
        source.disconnect();
    }
    if (processor) {
        processor.disconnect();
    }
}


async function startRecording() {
    console.log("Starting recording...");
    isRecording = true;
    recordButton.textContent = 'Stop Recording';
    recordButton.style.backgroundColor = '#ff4444';
    pauseButton.style.display = 'none'; // Hide pause button during recording
    
    if (!textHidden) {
        circleText.style.opacity = '0';
        textHidden = true;
    }
    
    animateWaves();

    socket = io({ transports: ['websocket'] });
    
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
        
        if (parsedData && parsedData.predicted_emotion && parsedData.emotions) {
            const mainEmotion = parsedData.predicted_emotion;
            const mainConfidence = parsedData.emotions[mainEmotion];
            
            emotionSpan.textContent = mainEmotion;
            confidenceSpan.textContent = mainConfidence.toFixed(2) + '%';
            
            emotionSpan.style.color = emotionColors[mainEmotion];

            const timestamp = new Date(parsedData.timestamp);
            const formattedTime = timestamp.toTimeString().split(' ')[0];

            emotionData.labels.push(formattedTime);
            
            emotions.forEach((emotion, index) => {
                emotionData.datasets[index].data.push(parsedData.emotions[emotion]);
            });
            
            if (emotionData.labels.length > maxDataPoints) {
                emotionData.labels.shift();
                emotionData.datasets.forEach(dataset => dataset.data.shift());
            }
            
            emotionChart.update();
        } else {
            console.error('Invalid emotion result data:', parsedData);
        }
    });

    try {
        console.log("Requesting microphone access...");
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        console.log("Microphone access granted");
        
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
    //emotionData.labels = [];
    //motionData.datasets.forEach(dataset => dataset.data = []);
    //emotionChart.update();
}

function discardAudio() {
    if (isPlaying) {
        stopPlayback();
    }
    
    audioBuffer = null;
    recordButton.textContent = 'Start Recording';
    recordButton.style.backgroundColor = '#8B7FD3';
    discardButton.style.display = 'none';
    uploadButton.style.display = 'inline-block';
    pauseButton.style.display = 'none';
    
    // Reset the file input
    fileInput.value = '';
}

function handleEmotionResult(data) {
    console.log('Received emotion result:', data);
    let parsedData;
    try {
        parsedData = typeof data === 'string' ? JSON.parse(data) : data;
    } catch (error) {
        console.error('Error parsing emotion result:', error);
        return;
    }
    
    if (parsedData && parsedData.predicted_emotion && parsedData.emotions) {
        const mainEmotion = parsedData.predicted_emotion;
        const mainConfidence = parsedData.emotions[mainEmotion];
        
        emotionSpan.textContent = mainEmotion;
        confidenceSpan.textContent = mainConfidence.toFixed(2) + '%';
        emotionSpan.style.color = emotionColors[mainEmotion];

        const timestamp = new Date(parsedData.timestamp);
        const formattedTime = timestamp.toTimeString().split(' ')[0];
        
        emotionData.labels.push(formattedTime);
        
        emotions.forEach((emotion, index) => {
            emotionData.datasets[index].data.push(parsedData.emotions[emotion]);
        });
        
        if (emotionData.labels.length > maxDataPoints) {
            emotionData.labels.shift();
            emotionData.datasets.forEach(dataset => dataset.data.shift());
        }
        
        emotionChart.update();
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

document.addEventListener('DOMContentLoaded', initChart);
console.log("script.js loaded");
