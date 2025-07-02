import * as WebRTC from "./webrtc";
import * as WebSocket from "./websocket";
import { GaussianAvatar } from './gaussianAvatar';

const assetPath = './asset/arkit/p2-1.zip';


// render
const div = document.getElementById('LAM_WebRender');
const gaussianAvatar = new GaussianAvatar(div as HTMLDivElement, assetPath);
gaussianAvatar.start();

// 将GaussianAvatar实例传递给websocket模块
WebSocket.setAvatarInstance(gaussianAvatar);

const audioEl = document.getElementById("audio-el") as HTMLAudioElement;
const statusEl = document.getElementById("status") as HTMLElement;
const buttonEl = document.getElementById("connect-btn") as HTMLButtonElement;
const serverUrl = document.getElementById("serverUrl") as HTMLInputElement;
const wsUrl = document.getElementById("wsUrl") as HTMLInputElement;

if (!audioEl || !statusEl || !buttonEl || !serverUrl || !wsUrl) {
    throw new Error("Required DOM elements not found");
}

let rtc_connected = false;
let ws_connected = false;
let peerConnection: RTCPeerConnection | null = null;

function generateShortUUID(): string {
    const chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let result = "";
    // Generate 22 characters (similar to short-uuid's default length)
    for (let i = 0; i < 22; i++) {
        result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
}

const _onConnecting = (): void => {
    statusEl.textContent = "WebRTC Connecting";
    buttonEl.textContent = "WebRTC Disconnect";
    rtc_connected = false;
};

const _onConnected = (): void => {
    if (!ws_connected) {
        statusEl.textContent = "WebRTC Connected, wait WebSocket connection";
        buttonEl.textContent = "WebRTC Disconnect";
    } else {
        statusEl.textContent =
            "WebRTC/WebSocket Connected, Please talk with chatbot";
        buttonEl.textContent = "Disconnect";
    }
    rtc_connected = true;
};

const _onDisconnected = (): void => {
    if (!ws_connected) {
        statusEl.textContent = "Disconnected";
        buttonEl.textContent = "Connect";
    } else {
        statusEl.textContent = "WebRTC Disconnected";
        buttonEl.textContent = "WebRTC Connect";
    }
    rtc_connected = false;
    ws_connected = false;
    // gaussianAvatar render speaking -> Idle have some bug :)
    gaussianAvatar.updateAvatarStatus("Idle");
};

const _onWSOpening = (): void => {
    statusEl.textContent = "WebSocket Connecting";
    buttonEl.textContent = "WebSocket Disconnect";
    rtc_connected = false;
};

const _onWSOpen = (): void => {
    if (!rtc_connected) {
        statusEl.textContent = "WebSocket Connected, wait WebRTC connection";
        buttonEl.textContent = "WebSocket Disconnect";
    } else {
        statusEl.textContent =
            "WebRTC/WebSocket Connected, Please talk with chatbot";
        buttonEl.textContent = "Disconnect";
    }
    ws_connected = true;
};

const _onWSClose = (): void => {
    if (!rtc_connected) {
        statusEl.textContent = "Disconnected";
        buttonEl.textContent = "Connect";
    } else {
        statusEl.textContent = "WebSocket Disconnected";
        buttonEl.textContent = "WebSocket Connect";
    }
    ws_connected = false;
    // gaussianAvatar render speaking -> Idle have some bug :)
    gaussianAvatar.updateAvatarStatus("Idle");
};

const _onTrack = (e: RTCTrackEvent): void => {
    console.log('Received remote stream:', e.streams[0]);
    // remote audio [0] track
    //audioEl.srcObject = e.streams[0];
};

interface MediaDevicesWithSampleRate extends MediaDevices {
    getUserMedia(constraints: MediaStreamConstraints & {
        sampleRate?: number;
        channelCount?: number;
        autoGainControl?: boolean;
        echoCancellation?: boolean;
        noiseSuppression?: boolean;
    }): Promise<MediaStream>;
}

const connect = async (): Promise<void> => {
    _onConnecting();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("getUserMedia is not supported in your browser.");
        return;
    }

    // Caller Open MediaStream (audio only)
    const audioStream = await (navigator.mediaDevices as MediaDevicesWithSampleRate).getUserMedia({
        audio: true,
        sampleRate: WebSocket.SAMPLE_RATE,
        channelCount: WebSocket.NUM_CHANNELS,
        autoGainControl: true,
        echoCancellation: true,
        noiseSuppression: true,
    });

    const peerID = generateShortUUID();

    // Caller Create webrtc peer connection with audio stream
    peerConnection = await WebRTC.createSmallWebRTCConnection(
        audioStream,
        serverUrl.value,
        peerID,
        _onConnected,
        _onDisconnected,
        _onTrack
    );

    _onWSOpening();
    // connect websocket
    WebSocket.startAudio(wsUrl.value + "/" + peerID, _onWSOpen, _onWSClose);
};

const disconnect = (): void => {
    if (rtc_connected) {
        _onDisconnected();
    }
    if (peerConnection) {
        peerConnection.close();
        peerConnection = null;
    }

    if (ws_connected) {
        WebSocket.stopAudio(true);
        _onWSClose();
    }
    statusEl.textContent = "Disconnected";
    buttonEl.textContent = "Connect";
};

buttonEl.addEventListener("click", async () => {
    if (!rtc_connected && !ws_connected) {
        await connect();
    } else {
        disconnect();
    }
});