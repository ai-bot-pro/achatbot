import * as WebRTC from "./webrtc.js";
import * as WebSocket from "./websocket.js";

const audioEl = document.getElementById("audio-el");
const statusEl = document.getElementById("status");
const buttonEl = document.getElementById("connect-btn");
const serverUrl = document.getElementById("serverUrl");
const wsUrl = document.getElementById("wsUrl");

let rtc_connected = false;
let ws_connected = false;
let peerConnection = null;

function generateShortUUID() {
  const chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
  let result = "";
  // Generate 22 characters (similar to short-uuid's default length)
  for (let i = 0; i < 22; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}
const _onConnecting = () => {
  statusEl.textContent = "WebRTC Connecting";
  buttonEl.textContent = "WebRTC Disconnect";
  rtc_connected = false;
};

const _onConnected = () => {
  if (!ws_connected) {
    statusEl.textContent = "WebRTC Connected, wait WebsScket connection";
    buttonEl.textContent = "WebRTC Disconnect";
  } else {
    statusEl.textContent =
      "WebRTC/WebSocket Connected, Please talk with chatbot";
    buttonEl.textContent = "Disconnect";
  }
  rtc_connected = true;
};

const _onDisconnected = () => {
  if (!ws_connected) {
    statusEl.textContent = "Disconnected";
    buttonEl.textContent = "Connect";
  } else {
    statusEl.textContent = "WebRTC Disconnected";
    buttonEl.textContent = "WebRTC Connect";
  }
  rtc_connected = false;
};

const _onWSOpening = () => {
  statusEl.textContent = "WebSocket Connecting";
  buttonEl.textContent = "WebSocket Disconnect";
  rtc_connected = false;
};
const _onWSOpen = () => {
  if (!rtc_connected) {
    statusEl.textContent = "WebsScket Connected, wait WebRTC connection";
    buttonEl.textContent = "WebSocket Disconnect";
  } else {
    statusEl.textContent =
      "WebRTC/WebSocket Connected, Please talk with chatbot";
    buttonEl.textContent = "Disconnect";
  }
  ws_connected = true;
};
const _onWSClose = () => {
  if (!rtc_connected) {
    statusEl.textContent = "Disconnected";
    buttonEl.textContent = "Connect";
  } else {
    statusEl.textContent = "WebSocket Disconnected";
    buttonEl.textContent = "WebSocket Connect";
  }
  ws_connected = false;
};

const _onTrack = (e) => {
  audioEl.srcObject = e.streams[0];
};

const connect = async () => {
  _onConnecting();

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("getUserMedia is not supported in your browser.");
    return;
  }

  // Caller Open MediaStream (audio only)
  const audioStream = await navigator.mediaDevices.getUserMedia({
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

const disconnect = () => {
  if (rtc_connected) {
    _onDisconnected();
  }
  if (peerConnection) {
    peerConnection.close();
    peerConnection = null;
  }

  if (ws_connected) {
    WebSocket.stopAudio();
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
