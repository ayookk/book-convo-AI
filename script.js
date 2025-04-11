const recordButton = document.getElementById('record');
const stopButton = document.getElementById('stop');
const timerDisplay = document.getElementById('timer');

let mediaRecorder;
let audioChunks = [];
let timerInterval;
let startTime;
let elapsedTime = 0;
let recordingStream;

function formatTime(time) {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

// AudioContext for visualizing audio
let audioContext;
let analyser;
let dataArray;
const canvasElement = document.getElementById('visualizer');
let canvasCtx;
let animationId;

// Initialize visualizer if canvas exists
if (canvasElement) {
    canvasCtx = canvasElement.getContext('2d');
    
    // Set canvas dimensions
    canvasElement.width = canvasElement.clientWidth;
    canvasElement.height = canvasElement.clientHeight;
}

function initializeAudioVisualizer(stream) {
    if (!canvasElement) return;
    
    // Create audio context and analyzer
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    
    source.connect(analyser);
    
    // Get frequency data
    const bufferLength = analyser.frequencyBinCount;
    dataArray = new Uint8Array(bufferLength);
    
    // Start drawing
    drawVisualizer();
}

function drawVisualizer() {
    if (!canvasElement || !analyser) return;
    
    // Request next animation frame
    animationId = requestAnimationFrame(drawVisualizer);
    
    // Get frequency data
    analyser.getByteFrequencyData(dataArray);
    
    // Clear canvas
    canvasCtx.fillStyle = 'rgb(200, 200, 200)';
    canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Draw bars
    const barWidth = (canvasElement.width / dataArray.length) * 2.5;
    let barHeight;
    let x = 0;
    
    for (let i = 0; i < dataArray.length; i++) {
        barHeight = dataArray[i] / 2;
        
        // Create gradient
        const gradient = canvasCtx.createLinearGradient(0, 0, 0, canvasElement.height);
        gradient.addColorStop(0, 'rgb(0, 123, 255)');
        gradient.addColorStop(1, 'rgb(0, 83, 172)');
        
        canvasCtx.fillStyle = gradient;
        canvasCtx.fillRect(x, canvasElement.height - barHeight, barWidth, barHeight);
        
        x += barWidth + 1;
    }
}

function stopVisualizer() {
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
    
    if (audioContext) {
        audioContext.close().catch(console.error);
        audioContext = null;
        analyser = null;
    }
    
    // Clear canvas
    if (canvasElement && canvasCtx) {
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    }
}

// Check if buttons exist before adding event listeners
if (recordButton && stopButton) {
    // Add event listeners to record and stop buttons
    recordButton.addEventListener('click', () => {
        // Reset audio chunks array
        audioChunks = [];
        
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                recordingStream = stream;
                
                // Options for better audio quality (higher bitrate)
                const options = { 
                    mimeType: 'audio/webm;codecs=opus',
                    audioBitsPerSecond: 128000
                };
                
                try {
                    mediaRecorder = new MediaRecorder(stream, options);
                } catch (e) {
                    // Fallback if the specified options aren't supported
                    console.warn('MediaRecorder with specified options not supported, using default codec', e);
                    mediaRecorder = new MediaRecorder(stream);
                }
                
                mediaRecorder.start();
                
                // Initialize audio visualizer
                if (canvasElement) {
                    initializeAudioVisualizer(stream);
                }
                
                startTime = Date.now() - elapsedTime;
                timerInterval = setInterval(() => {
                    elapsedTime = Math.floor((Date.now() - startTime) / 1000);
                    timerDisplay.textContent = formatTime(elapsedTime);
                }, 1000);
                
                mediaRecorder.ondataavailable = e => {
                    if (e.data.size > 0) {
                        audioChunks.push(e.data);
                    }
                };
                
                mediaRecorder.onstop = () => {
                    clearInterval(timerInterval);
                    
                    // Stop visualizer
                    stopVisualizer();
                    
                    // Reset timer display
                    elapsedTime = 0;
                    timerDisplay.textContent = '0:00';
                    
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    
                    // Create a loading indicator
                    const loadingIndicator = document.createElement('div');
                    loadingIndicator.id = 'loadingIndicator';
                    loadingIndicator.textContent = 'Processing your question and generating response...';
                    loadingIndicator.style.position = 'fixed';
                    loadingIndicator.style.top = '50%';
                    loadingIndicator.style.left = '50%';
                    loadingIndicator.style.transform = 'translate(-50%, -50%)';
                    loadingIndicator.style.padding = '20px';
                    loadingIndicator.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
                    loadingIndicator.style.color = 'white';
                    loadingIndicator.style.borderRadius = '5px';
                    loadingIndicator.style.zIndex = '1000';
                    document.body.appendChild(loadingIndicator);
                    
                    const formData = new FormData();
                    formData.append('audio_data', audioBlob, 'recorded_audio.wav');
                    
                    fetch('/upload_audio', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Upload failed');
                        }
                        window.location.reload();
                    })
                    .catch(error => {
                        document.body.removeChild(loadingIndicator);
                        console.error('Error uploading audio:', error);
                        alert('Failed to upload audio. Please try again.');
                    })
                    .finally(() => {
                        // If for some reason the page didn't reload, remove the loading indicator
                        if (document.getElementById('loadingIndicator')) {
                            document.body.removeChild(loadingIndicator);
                        }
                    });
                    
                    // Stop all tracks in the stream
                    if (recordingStream) {
                        recordingStream.getTracks().forEach(track => track.stop());
                        recordingStream = null;
                    }
                };
            })
            .catch(error => {
                console.error('Error accessing microphone:', error);
                alert('Unable to access microphone. Please ensure microphone permissions are granted.');
                recordButton.disabled = false;
                stopButton.disabled = true;
            });
        
        recordButton.disabled = true;
        stopButton.disabled = false;
    });

    stopButton.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        
        // Clear interval for timer
        clearInterval(timerInterval);
        
        // Stop the stream tracks
        if (recordingStream) {
            recordingStream.getTracks().forEach(track => track.stop());
            recordingStream = null;
        }
        
        // Stop visualizer
        stopVisualizer();
        
        recordButton.disabled = false;
        stopButton.disabled = true;
    });

    // Initial button state
    stopButton.disabled = true;
}

// Handle page visibility change to prevent memory leaks
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        
        clearInterval(timerInterval);
        
        if (recordingStream) {
            recordingStream.getTracks().forEach(track => track.stop());
            recordingStream = null;
        }
        
        stopVisualizer();
        
        if (recordButton && stopButton) {
            recordButton.disabled = false;
            stopButton.disabled = true;
        }
    }
});