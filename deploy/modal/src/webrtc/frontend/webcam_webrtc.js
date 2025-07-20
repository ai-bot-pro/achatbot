import { WebRtcClient } from './webrtc.js';

// Add status display element
const statusDisplay = document.getElementById('statusDisplay');
const MAX_STATUS_HISTORY = 100;
let statusHistory = [];

// Audio visualizer class
class AudioVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.canvasCtx = canvas.getContext('2d');
        this.audioCtx = null;
        this.analyser = null;
        this.dataArray = null;
        this.animationId = null;
        this.isActive = false;
    }

    setup(stream) {
        if (!stream) return;
        
        // Create audio context if it doesn't exist
        if (!this.audioCtx) {
            this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        // Create analyser
        this.analyser = this.audioCtx.createAnalyser();
        this.analyser.fftSize = 256;
        const bufferLength = this.analyser.frequencyBinCount;
        this.dataArray = new Uint8Array(bufferLength);
        
        // Connect stream to analyser
        const source = this.audioCtx.createMediaStreamSource(stream);
        source.connect(this.analyser);
        
        // Start visualization
        this.isActive = true;
        this.draw();
    }

    draw() {
        if (!this.isActive) return;
        
        // Request next animation frame
        this.animationId = requestAnimationFrame(() => this.draw());
        
        // Get audio data
        this.analyser.getByteFrequencyData(this.dataArray);
        
        // Clear canvas
        const width = this.canvas.width;
        const height = this.canvas.height;
        this.canvasCtx.clearRect(0, 0, width, height);
        
        // Draw visualization
        const barWidth = (width / this.dataArray.length) * 2.5;
        let x = 0;
        
        this.canvasCtx.fillStyle = 'rgb(0, 200, 255)';
        
        for (let i = 0; i < this.dataArray.length; i++) {
            const barHeight = (this.dataArray[i] / 255) * height;
            this.canvasCtx.fillRect(x, height - barHeight, barWidth, barHeight);
            x += barWidth + 1;
        }
    }

    stop() {
        this.isActive = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        // Clear canvas
        if (this.canvas && this.canvasCtx) {
            this.canvasCtx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }
}


// DOM elements
const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');
const localAudioVisualizer = document.getElementById('localAudioVisualizer');
const remoteAudioVisualizer = document.getElementById('remoteAudioVisualizer');
const startWebcamButton = document.getElementById('startWebcamButton');
const startStreamingButton = document.getElementById('startStreamingButton');
const stopStreamingButton = document.getElementById('stopStreamingButton');

// Initialize WebRTC client and audio visualizers
const webrtcClient = new WebRtcClient();
const localVisualizer = new AudioVisualizer(localAudioVisualizer);
const remoteVisualizer = new AudioVisualizer(remoteAudioVisualizer);

// Set up event listeners
webrtcClient.addEventListener('status', (event) => {
    // Add timestamp to message
    const now = new Date();
    const timestamp = now.toLocaleTimeString();
    const statusLine = `[${timestamp}] ${event.detail.message}`;
    
    // Add to history
    statusHistory.push(statusLine);
    
    // Keep only last MAX_STATUS_HISTORY messages
    if (statusHistory.length > MAX_STATUS_HISTORY) {
        statusHistory.shift();
    }
    
    // Update display
    statusDisplay.innerHTML = statusHistory.map(line => 
        `<div class="status-line">${line}</div>`
    ).join('');
    
    // Scroll to bottom
    statusDisplay.scrollTop = statusDisplay.scrollHeight;
});

webrtcClient.addEventListener('localStream', (event) => {
    localVideo.srcObject = event.detail.stream;
    // Setup local audio visualizer
    localVisualizer.setup(event.detail.stream);
});

webrtcClient.addEventListener('remoteStream', (event) => {
    remoteVideo.srcObject = event.detail.stream;
    // Setup remote audio visualizer
    remoteVisualizer.setup(event.detail.stream);
});

webrtcClient.addEventListener('error', (event) => {
    console.error('WebRTC error:', event.detail.error);
});

webrtcClient.addEventListener('connectionStateChange', (event) => {
    if (event.detail.state === 'connected') {
        startStreamingButton.disabled = true;
        stopStreamingButton.disabled = false;
    }
});

webrtcClient.addEventListener('streamingStopped', () => {
    stopStreamingButton.disabled = true;
    startStreamingButton.disabled = false;
    remoteVideo.srcObject = null;
    // Stop remote audio visualizer
    remoteVisualizer.stop();
});

// Initialize button states
startWebcamButton.disabled = false;
startStreamingButton.disabled = true;
stopStreamingButton.disabled = true;

// Event handlers
async function handleStartWebcam() {
    try {
        await webrtcClient.startWebcam();
        startWebcamButton.disabled = true;
        startStreamingButton.disabled = false;
    } catch (err) {
        console.error('Error starting webcam:', err);
    }
}

async function handleStartStreaming() {
    try {
        // Get WebSocket server URL
        const wsServerUrl = document.getElementById('wsServer').value.trim();
        webrtcClient.setWsServerUrl(wsServerUrl);
        
        startWebcamButton.disabled = true;
        startStreamingButton.disabled = true;
        stopStreamingButton.disabled = false;
        await webrtcClient.startStreaming();
    } catch (err) {
        console.error('Error starting streaming:', err);
        webrtcClient.dispatchEvent(new CustomEvent('status', {
            detail: { message: `Error: ${err.message}` }
        }));
        startStreamingButton.disabled = false;
    }
}

async function handleStopStreaming() {
    await webrtcClient.stopStreaming();
    // Stop both visualizers
    localVisualizer.stop();
    remoteVisualizer.stop();
}

// Add event listener for STUN/TURN server radio buttons
document.querySelectorAll('input[name="iceServer"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        webrtcClient.setIceServerType(e.target.value);
    });
});

// Event listeners
startWebcamButton.addEventListener('click', handleStartWebcam);
startStreamingButton.addEventListener('click', handleStartStreaming);
stopStreamingButton.addEventListener('click', handleStopStreaming);

// Add cleanup handler for when browser tab is closed
window.addEventListener('beforeunload', async () => {
    await webrtcClient.cleanup();
    // Stop audio visualizers
    localVisualizer.stop();
    remoteVisualizer.stop();
    // ensure stun/turn radio and iceServerType are reset
    document.querySelectorAll('input[name="iceServer"]').forEach(radio => {
        if (radio.value == "turn") {
            radio.checked = false;
        } else {
            radio.checked = true;
        }
    });
    webrtcClient.setIceServerType('stun');
});