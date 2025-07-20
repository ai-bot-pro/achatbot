import * as WebRTC from "./webrtc";
import * as WebSocket from "./websocket";
import { GaussianAvatar } from './gaussianAvatar';

const assetPath = import.meta.env.VITE_AVATAR_PATH || './assets/arkit/me.zip';
//const assetPath = './asset/arkit/p2-1.zip';

// 从环境变量获取服务器 URL
const DEFAULT_SERVER_URL = import.meta.env.VITE_SERVER_URL || 'http://localhost:4321';

// render
const div = document.getElementById('LAM_WebRender');
const statusEl = document.getElementById("status") as HTMLElement;
const gaussianAvatar = new GaussianAvatar(div as HTMLDivElement, assetPath);

// 在启动前更新状态
if (statusEl) {
    statusEl.textContent = "Loading Avatar...";
}

// 启动头像渲染
gaussianAvatar.start();

// 将GaussianAvatar实例传递给websocket模块
WebSocket.setAvatarInstance(gaussianAvatar);

// 检查加载状态并更新UI
const checkLoadingStatus = () => {
    if (gaussianAvatar.isLoading()) {
        if (statusEl) {
            statusEl.textContent = "Loading Avatar...";
        }
        // 继续检查
        setTimeout(checkLoadingStatus, 500);
    } else {
        // 加载完成，恢复默认状态
        if (statusEl && statusEl.textContent === "Loading Avatar...") {
            statusEl.textContent = "Disconnected";
        }
    }
};

// 开始检查加载状态
checkLoadingStatus();

const audioEl = document.getElementById("audio-el") as HTMLAudioElement;
const buttonEl = document.getElementById("connect-btn") as HTMLButtonElement;
const serverUrl = document.getElementById("serverUrl") as HTMLInputElement;
const wsUrl = document.getElementById("wsUrl") as HTMLInputElement;

if (!audioEl || !statusEl || !buttonEl || !serverUrl || !wsUrl) {
    throw new Error("Required DOM elements not found");
}

// 设置默认服务器 URL
serverUrl.value = DEFAULT_SERVER_URL;

let rtc_connected = false;
let ws_connected = false;
let connecting = false; // 新增状态变量，表示是否正在连接中
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
        // 当WebRTC和WebSocket都连接成功时，重置连接状态
        connecting = false;
        buttonEl.disabled = false;
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
    // 重置连接状态并启用按钮
    connecting = false;
    buttonEl.disabled = false;
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
        // 当WebRTC和WebSocket都连接成功时，重置连接状态
        connecting = false;
        buttonEl.disabled = false;
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
    // 设置连接状态为正在连接中
    connecting = true;
    // 禁用按钮，防止重复点击
    buttonEl.disabled = true;

    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert("getUserMedia is not supported in your browser.");
            throw new Error("getUserMedia not supported");
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

        _onWSOpening();
        let websocketBaseUrl = serverUrl.value.trim();
        if (websocketBaseUrl.startsWith('http://')) {
            websocketBaseUrl = websocketBaseUrl.replace('http://', 'ws://');
        } else if (websocketBaseUrl.startsWith('https://')) {
            websocketBaseUrl = websocketBaseUrl.replace('https://', 'wss://');
        } else if (websocketBaseUrl.startsWith('ws://')) {
        } else if (websocketBaseUrl.startsWith('wss://')) {
        } else {
            // 如果没有协议前缀，默认添加 ws://
            websocketBaseUrl = 'ws://' + websocketBaseUrl;
        }
        wsUrl.value = websocketBaseUrl;

        const websocketUrl = new URL(`/${peerID}`, websocketBaseUrl).toString();
        console.log("Connecting to WebSocket server:", websocketUrl);
        WebSocket.startAudio(websocketUrl, _onWSOpen, _onWSClose);

        _onConnecting();

        // Caller Create webrtc peer connection with audio stream
        peerConnection = await WebRTC.createSmallWebRTCConnection(
            audioStream,
            serverUrl.value,
            peerID,
            _onConnected,
            _onDisconnected,
            _onTrack
        );

    } catch (error) {
        console.error("Connection failed:", error);
        // 连接失败时重置连接状态并启用按钮
        connecting = false;
        buttonEl.disabled = false;
        statusEl.textContent = "Connection failed";
        buttonEl.textContent = "Connect";
    }
};

const disconnect = (): void => {
    // 标记为用户主动断开连接
    const userInitiated = true;

    if (rtc_connected) {
        _onDisconnected();
    }
    if (peerConnection) {
        peerConnection.close();
        peerConnection = null;
    }

    if (ws_connected) {
        // 传递userInitiated参数，确保WebSocket模块知道这是用户主动断开
        WebSocket.stopAudio(true, userInitiated);
        _onWSClose();
    }
    statusEl.textContent = "Disconnected";
    buttonEl.textContent = "Connect";

    // 重置连接状态并启用按钮
    connecting = false;
    buttonEl.disabled = false;
};

buttonEl.addEventListener("click", async () => {
    // 如果正在连接中，不执行任何操作
    if (connecting) {
        return;
    }

    if (!rtc_connected && !ws_connected) {
        await connect();
    } else {
        disconnect();
    }
});