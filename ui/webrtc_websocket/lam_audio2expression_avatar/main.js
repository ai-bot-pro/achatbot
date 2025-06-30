import * as WebRTC from "./webrtc.js";
import * as WebSocket from "./websocket.js";

const audioEl = document.getElementById("audio-el");
const statusEl = document.getElementById("status");
const buttonEl = document.getElementById("connect-btn");
const serverUrl = document.getElementById("serverUrl");
const wsUrl = document.getElementById("wsUrl");

let connected = false;
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
  statusEl.textContent = "Connecting";
  buttonEl.textContent = "Disconnect";
  connected = true;
};

const _onConnected = () => {
  statusEl.textContent = "Connected, Please talk with chatbot";
  buttonEl.textContent = "Disconnect";
  connected = true;
};

const _onDisconnected = () => {
  statusEl.textContent = "Disconnected";
  buttonEl.textContent = "Connect";
  connected = false;
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

  // connect websocket
  WebSocket.startAudio(wsUrl.value + "/" + peerID);
};

const disconnect = () => {
  if (!peerConnection) {
    return;
  }
  peerConnection.close();
  peerConnection = null;

  WebSocket.stopAudio();

  _onDisconnected();
};

buttonEl.addEventListener("click", async () => {
  if (!connected) {
    await connect();
  } else {
    disconnect();
  }
});
