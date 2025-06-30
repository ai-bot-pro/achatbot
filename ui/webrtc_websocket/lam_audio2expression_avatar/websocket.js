export const SAMPLE_RATE = 16000;
export const SAMPLE_WIDTH = 2;
export const NUM_CHANNELS = 1;

const PLAY_TIME_RESET_THRESHOLD_MS = 1.0;

// The protobuf type. We will load it later.
let Frame = null;

// The websocket connection.
let ws = null;

// The audio context
let audioContext = null;

// The audio context media stream source
let source = null;

// The microphone stream from getUserMedia. SHould be sampled to the
// proper sample rate.
let microphoneStream = null;

// Script processor to get data from microphone.
let scriptProcessor = null;

// AudioContext play time.
let playTime = 0;

// Last time we received a websocket message.
let lastMessageTime = 0;

// Whether we should be playing audio.
let isPlaying = false;

const proto = protobuf.load("avatar_data_frames.proto", (err, root) => {
  if (err) {
    throw err;
  }
  Frame = root.lookupType("achatbot_frames.Frame");
  console.log("Loaded protobuf");
});

export function startAudio(wsUrl) {
  audioContext = new (window.AudioContext || window.webkitAudioContext)({
    latencyHint: "interactive",
    sampleRate: SAMPLE_RATE,
  });

  if (wsUrl) {
    initWebSocket(wsUrl);
  }

  isPlaying = true;
}

function initWebSocket(wsUrl) {
  console.log("Connecting to websocket server:", wsUrl);
  ws = new WebSocket(wsUrl);

  ws.addEventListener("open", () =>
    console.log("WebSocket connection established.")
  );
  ws.addEventListener("message", handleWebSocketMessage);
  ws.addEventListener("close", (event) => {
    console.log("WebSocket connection closed.", event.code, event.reason);
    stopAudio(false);
  });
  ws.addEventListener("error", (event) =>
    console.error("WebSocket error:", event)
  );
}

async function handleWebSocketMessage(event) {
  const arrayBuffer = await event.data.arrayBuffer();
  if (isPlaying) {
    enqueueAudioFromProto(arrayBuffer);
  }
}

function enqueueAudioFromProto(arrayBuffer) {
  const parsedFrame = Frame.decode(new Uint8Array(arrayBuffer));
  if (!parsedFrame?.animation_audio) {
    return false;
  }

  // Reset play time if it's been a while we haven't played anything.
  const diffTime = audioContext.currentTime - lastMessageTime;
  if (playTime == 0 || diffTime > PLAY_TIME_RESET_THRESHOLD_MS) {
    playTime = audioContext.currentTime;
  }
  lastMessageTime = audioContext.currentTime;

  // We should be able to use parsedFrame.audio.audio.buffer but for
  // some reason that contains all the bytes from the protobuf message.
  const audioVector = Array.from(parsedFrame.audio.audio);
  const audioArray = new Uint8Array(audioVector);

  audioContext.decodeAudioData(audioArray.buffer, function (buffer) {
    const source = new AudioBufferSourceNode(audioContext);
    source.buffer = buffer;
    source.start(playTime);
    source.connect(audioContext.destination);
    playTime = playTime + buffer.duration;
  });
}

export function stopAudio(closeWebsocket) {
  playTime = 0;
  isPlaying = false;

  if (ws && closeWebsocket) {
    ws.close();
    ws = null;
  }

  if (scriptProcessor) {
    scriptProcessor.disconnect();
  }
  if (source) {
    source.disconnect();
  }
}
