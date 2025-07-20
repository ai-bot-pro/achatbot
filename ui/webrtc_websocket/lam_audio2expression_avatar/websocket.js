export const SAMPLE_RATE = 16000;
export const SAMPLE_WIDTH = 2;
export const NUM_CHANNELS = 1;

const PLAY_TIME_RESET_THRESHOLD_MS = 1;

// The protobuf type. We will load it later.
let Frame = null;

// The websocket connection.
let ws = null;

// Heartbeat interval ID
let heartbeatInterval = null;

// Reconnection settings
const RECONNECT_DELAY = 3000; // 3 seconds
const MAX_RECONNECT_ATTEMPTS = 5;
let reconnectAttempts = 0;
let reconnectTimeout = null;

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

export function startAudio(wsUrl, _OnOpen, _OnClose) {
  audioContext = new (window.AudioContext || window.webkitAudioContext)({
    latencyHint: "interactive",
    sampleRate: SAMPLE_RATE,
  });

  if (wsUrl) {
    initWebSocket(wsUrl, _OnOpen, _OnClose);
  }

  isPlaying = true;
}

function initWebSocket(wsUrl, _OnOpen, _OnClose) {
  console.log("Connecting to websocket server:", wsUrl);

  // Clear any existing heartbeat interval
  if (heartbeatInterval) {
    clearInterval(heartbeatInterval);
    heartbeatInterval = null;
  }

  // Clear any existing reconnect timeout
  if (reconnectTimeout) {
    clearTimeout(reconnectTimeout);
    reconnectTimeout = null;
  }

  ws = new WebSocket(wsUrl);

  ws.addEventListener("open", () => {
    _OnOpen();
    console.log("WebSocket connection established.");
    reconnectAttempts = 0; // Reset reconnect attempts on successful connection

    // Start heartbeat after connection is established
    heartbeatInterval = setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        // Send a ping message to keep the connection alive
        ws.send(JSON.stringify({ type: "ping" }));
      }
    }, 30000); // Send heartbeat every 30 seconds
  });

  ws.addEventListener("message", handleWebSocketMessage);

  ws.addEventListener("close", (event) => {
    _OnClose();
    console.log("WebSocket connection closed.", event.code, event.reason);
    stopAudio(false);

    // Clear heartbeat interval
    if (heartbeatInterval) {
      clearInterval(heartbeatInterval);
      heartbeatInterval = null;
    }

    // Attempt to reconnect if not closed intentionally
    if (event.code !== 1000 && reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
      console.log(
        `Attempting to reconnect (${
          reconnectAttempts + 1
        }/${MAX_RECONNECT_ATTEMPTS})...`
      );
      reconnectAttempts++;
      reconnectTimeout = setTimeout(() => {
        initWebSocket(wsUrl, _OnOpen, _OnClose);
      }, RECONNECT_DELAY);
    } else {
      _OnClose();
    }
  });

  ws.addEventListener("error", (event) => {
    console.error("WebSocket error:", event);
    // Error handling is managed by the close event handler
  });
}

async function handleWebSocketMessage(event) {
  try {
    // Check if the message is a text message (could be a pong response)
    if (typeof event.data === "string") {
      try {
        const jsonData = JSON.parse(event.data);
        if (jsonData.type === "pong") {
          console.log("Received heartbeat pong from server");
          return;
        }
      } catch (e) {
        // Not JSON, continue processing as binary
      }
    }

    // Process binary data (audio)
    const arrayBuffer = await event.data.arrayBuffer();
    if (isPlaying) {
      enqueueAudioFromProto(arrayBuffer);
    }
  } catch (error) {
    console.error("Error handling WebSocket message:", error);
  }
}

function enqueueAudioFromProto(arrayBuffer) {
  const parsedFrame = Frame.decode(new Uint8Array(arrayBuffer));
  if (!parsedFrame?.animationAudio) {
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
  const audioVector = Array.from(parsedFrame.animationAudio.audio);
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

  // Clear heartbeat interval
  if (heartbeatInterval) {
    clearInterval(heartbeatInterval);
    heartbeatInterval = null;
  }

  // Clear reconnect timeout
  if (reconnectTimeout) {
    clearTimeout(reconnectTimeout);
    reconnectTimeout = null;
  }

  if (ws && closeWebsocket) {
    // Reset reconnect attempts to prevent automatic reconnection
    reconnectAttempts = MAX_RECONNECT_ATTEMPTS;
    ws.close(1000); // Normal closure
    ws = null;
  }

  if (scriptProcessor) {
    scriptProcessor.disconnect();
  }
  if (source) {
    source.disconnect();
  }
}
