import * as protobuf from 'protobufjs';
import { json } from 'stream/consumers';
import { GaussianAvatar } from './gaussianAvatar';

export const SAMPLE_RATE = 16000;
// 全局变量，用于存储GaussianAvatar实例的引用
let avatarInstance: GaussianAvatar | null = null;

// 设置GaussianAvatar实例的方法
export function setAvatarInstance(avatar: GaussianAvatar): void {
    avatarInstance = avatar;
}
export const SAMPLE_WIDTH = 2;
export const NUM_CHANNELS = 1;

const PLAY_TIME_RESET_THRESHOLD_MS = 1;

// Protobuf type interfaces
interface Frame {
    animationAudio?: {
        audio: Uint8Array;
        sampleRate: number;
        numChannels: number;
        sampleWidth: number;
        animationJson: string;
        avatarStatus: string;
    };
}

type FrameType = protobuf.Type;

// The protobuf type. We will load it later.
let Frame: FrameType | null = null;

// The websocket connection.
let ws: WebSocket | null = null;

// Heartbeat interval ID
let heartbeatInterval: ReturnType<typeof setInterval> | null = null;


// Reconnection settings
const RECONNECT_DELAY = 3000; // 3 seconds
const MAX_RECONNECT_ATTEMPTS = 5;
let reconnectAttempts = 0;
let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;


// The audio context
let audioContext: AudioContext | null = null;

// The audio context media stream source
let source: MediaStreamAudioSourceNode | null = null;

// The microphone stream from getUserMedia. Should be sampled to the
// proper sample rate.
let microphoneStream: MediaStream | null = null;

// Script processor to get data from microphone.
let scriptProcessor: ScriptProcessorNode | null = null;

// AudioContext play time.
let playTime = 0;

// Last time we received a websocket message.
let lastMessageTime = 0;

// Whether we should be playing audio.
let isPlaying = false;

const proto = protobuf.load("../asset/avatar_data_frames.proto", (err: Error | null, root?: protobuf.Root) => {
    if (err || !root) {
        throw err || new Error('Failed to load protobuf root');
    }
    Frame = root.lookupType("achatbot_frames.Frame");
    console.log("Loaded protobuf");
});

export function startAudio(wsUrl: string, onOpen: () => void, onClose: () => void): void {
    const AudioContextConstructor = window.AudioContext || (window as any).webkitAudioContext;
    audioContext = new AudioContextConstructor({
        latencyHint: "interactive",
        sampleRate: SAMPLE_RATE,
    });

    if (wsUrl) {
        initWebSocket(wsUrl, onOpen, onClose);
    }

    isPlaying = true;
}

function initWebSocket(wsUrl: string, onOpen: () => void, onClose: () => void): void {
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
        onOpen();
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

    ws.addEventListener("close", (event: CloseEvent) => {
        onClose();
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
                `Attempting to reconnect (${reconnectAttempts + 1
                }/${MAX_RECONNECT_ATTEMPTS})...`
            );
            reconnectAttempts++;
            reconnectTimeout = setTimeout(() => {
                initWebSocket(wsUrl, onOpen, onClose);
            }, RECONNECT_DELAY);
        } else {
            onClose();
        }
    });

    ws.addEventListener("error", (event: Event) => {
        console.error("WebSocket error:", event);
        // Error handling is managed by the close event handler
    });
}

async function handleWebSocketMessage(event: MessageEvent): Promise<void> {
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

function enqueueAudioFromProto(arrayBuffer: ArrayBuffer): boolean {
    if (!Frame || !audioContext) {
        return false;
    }

    const parsedFrame = Frame.decode(new Uint8Array(arrayBuffer)) as Frame;
    if (!parsedFrame?.animationAudio) {
        return false;
    }

    const avatarStatus = parsedFrame.animationAudio?.avatarStatus;
    console.log("avatarStatus:", avatarStatus);
    // 将avatarStatus数据传递给GaussianAvatar实例
    if (avatarInstance && avatarStatus) {
        avatarInstance.updateAvatarStatus(avatarStatus);
    }

    const animationJsonStr = parsedFrame.animationAudio?.animationJson;
    if (animationJsonStr == null || animationJsonStr.length == 0 || animationJsonStr == "{}" || animationJsonStr == "[]") {
        // no animation, return 
        return true;
    }

    const animationJson = JSON.parse(animationJsonStr);
    console.log("Animation JSON:", animationJson);
    // 将animationJson数据传递给GaussianAvatar实例
    if (avatarInstance) {
        avatarInstance.updateAnimationData(animationJson);
    }

    if (parsedFrame.animationAudio.audio.length == 0) {
        // no audio,return 
        return true;
    }

    // Reset play time if it's been a while we haven't played anything.
    const diffTime = audioContext.currentTime - lastMessageTime;
    if (playTime === 0 || diffTime > PLAY_TIME_RESET_THRESHOLD_MS) {
        playTime = audioContext.currentTime;
    }
    lastMessageTime = audioContext.currentTime;

    // We should be able to use parsedFrame.audio.audio.buffer but for
    // some reason that contains all the bytes from the protobuf message.
    const audioVector = Array.from(parsedFrame.animationAudio.audio);
    const audioArray = new Uint8Array(audioVector);

    audioContext.decodeAudioData(audioArray.buffer, function (buffer) {
        if (!audioContext) return;

        const source = new AudioBufferSourceNode(audioContext);
        source.buffer = buffer;
        source.start(playTime);
        source.connect(audioContext.destination);
        playTime = playTime + buffer.duration;
    });

    return true;
}

export function stopAudio(closeWebsocket: boolean): void {
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