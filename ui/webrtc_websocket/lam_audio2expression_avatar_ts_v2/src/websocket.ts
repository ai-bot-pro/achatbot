import * as protobuf from 'protobufjs';
import { GaussianAvatar } from './gaussianAvatar';

const protoPath = import.meta.env.VITE_PROTOBUF_PATH || "./assets/avatar_data_frames.proto";

export const SAMPLE_RATE = 16000;
// 全局变量，用于存储GaussianAvatar实例的引用
let avatarInstance: GaussianAvatar | null = null;

// 设置GaussianAvatar实例的方法
export function setAvatarInstance(avatar: GaussianAvatar): void {
    avatarInstance = avatar;
    // 当音频上下文创建后，将其传递给头像实例
    if (audioContext && avatarInstance) {
        avatarInstance.setAudioContext(audioContext);
    }
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

// 添加一个Promise来跟踪WebSocket连接状态
let wsReadyPromise: Promise<void> | null = null;
let wsReadyResolve: (() => void) | null = null;
let wsReadyReject: ((reason?: any) => void) | null = null;

// 导出获取WebSocket连接的函数
export function getWebSocket(): WebSocket | null {
    return ws;
}

// 导出发送WebSocket消息的函数
export async function sendWebSocketMessage(message: any): Promise<void> {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        throw new Error("WebSocket connection is not open");
    }

    ws.send(JSON.stringify(message));
}

// 导出等待WebSocket连接就绪的函数
export function waitForWebSocketReady(): Promise<void> {
    if (ws && ws.readyState === WebSocket.OPEN) {
        return Promise.resolve();
    }

    if (!wsReadyPromise) {
        wsReadyPromise = new Promise<void>((resolve, reject) => {
            wsReadyResolve = resolve;
            wsReadyReject = reject;
        });
    }

    return wsReadyPromise;
}

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

// 音频队列，用于管理待播放的音频片段
interface AudioQueueItem {
    buffer: AudioBuffer;
    animationData: any;
    duration: number;
}
let audioQueue: AudioQueueItem[] = [];
let isProcessingQueue = false;

const proto = protobuf.load(protoPath, (err: Error | null, root?: protobuf.Root) => {
    if (err || !root) {
        throw err || new Error('Failed to load protobuf root');
    }
    Frame = root.lookupType("achatbot_frames.Frame");
    console.log("Loaded protobuf");
});

export function startAudio(wsUrl: string, onOpen: () => void, onClose: () => void): void {
    // 如果音频上下文已存在，尝试恢复它
    if (audioContext) {
        if (audioContext.state === 'suspended') {
            audioContext.resume().catch(err => {
                console.error("Error resuming audio context:", err);
                // 如果恢复失败，创建一个新的音频上下文
                createNewAudioContext();
            });
        }
    } else {
        // 创建新的音频上下文
        createNewAudioContext();
    }

    // 重置音频队列和播放状态
    audioQueue = [];
    isProcessingQueue = false;
    playTime = 0;

    // 确保初始状态是音频不播放
    if (avatarInstance) {
        avatarInstance.setAudioPlayingState(false);
    }

    if (wsUrl) {
        initWebSocket(wsUrl, onOpen, onClose);
    }

    isPlaying = true;
}

function createNewAudioContext(): void {
    const AudioContextConstructor = window.AudioContext || (window as any).webkitAudioContext;
    audioContext = new AudioContextConstructor({
        latencyHint: "interactive",
        sampleRate: SAMPLE_RATE,
    });

    // 添加状态变化事件监听器
    audioContext.addEventListener('statechange', handleAudioContextStateChange);

    console.log(`Created new audio context, state: ${audioContext.state}`);

    // 如果头像实例已经存在，将音频上下文传递给它
    if (avatarInstance && audioContext) {
        avatarInstance.setAudioContext(audioContext);
    }
}

// 处理音频上下文状态变化
function handleAudioContextStateChange(event: Event): void {
    if (!audioContext) return;

    console.log(`Audio context state changed to: ${audioContext.state}`);

    if (audioContext.state === 'running' && isPlaying && audioQueue.length > 0 && !isProcessingQueue) {
        console.log("Audio context is running, resuming queue processing");
        processAudioQueue();
    }
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
        console.log("WebSocket connection established.");

        // 重置重连尝试次数
        reconnectAttempts = 0;

        // 重置音频队列和播放状态
        audioQueue = [];
        isProcessingQueue = false;
        playTime = 0;

        // 确保初始状态是音频不播放
        if (avatarInstance) {
            avatarInstance.setAudioPlayingState(false);
        }

        // 确保音频上下文处于运行状态
        if (audioContext && audioContext.state === 'suspended') {
            console.log("Resuming audio context after WebSocket connection established");
            audioContext.resume().catch(err => {
                console.error("Error resuming audio context:", err);
            });
        }

        // 解析WebSocket就绪Promise
        if (wsReadyResolve) {
            wsReadyResolve();
            wsReadyPromise = null;
            wsReadyResolve = null;
            wsReadyReject = null;
        }

        // 通知应用程序连接已建立
        onOpen();

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
        console.log(`WebSocket connection closed. Code: ${event.code}, Reason: ${event.reason || 'No reason provided'}`);

        // 拒绝WebSocket就绪Promise
        if (wsReadyReject) {
            wsReadyReject(new Error(`WebSocket connection closed. Code: ${event.code}, Reason: ${event.reason || 'No reason provided'}`));
            wsReadyPromise = null;
            wsReadyResolve = null;
            wsReadyReject = null;
        }

        // 停止音频播放但不关闭WebSocket（因为它已经关闭了）
        stopAudio(false);

        // 通知应用程序连接已关闭
        onClose();

        // Clear heartbeat interval
        if (heartbeatInterval) {
            clearInterval(heartbeatInterval);
            heartbeatInterval = null;
        }

        // 检查是否是用户主动断开连接
        if (userInitiatedDisconnect) {
            console.log("User initiated disconnect, not attempting to reconnect");
            // 重置标志，以便将来的连接可以正常工作
            userInitiatedDisconnect = false;
            return;
        }

        // Attempt to reconnect if not closed intentionally
        if (event.code !== 1000 && reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            console.log(
                `Attempting to reconnect (${reconnectAttempts + 1}/${MAX_RECONNECT_ATTEMPTS}) in ${RECONNECT_DELAY}ms...`
            );
            reconnectAttempts++;
            reconnectTimeout = setTimeout(() => {
                console.log("Executing reconnection attempt...");
                initWebSocket(wsUrl, onOpen, onClose);
            }, RECONNECT_DELAY);
        } else if (event.code !== 1000) {
            console.log("Maximum reconnection attempts reached or connection closed intentionally");
        }
    });

    ws.addEventListener("error", (event: Event) => {
        console.error("WebSocket error occurred:", event);

        // 记录更多的连接状态信息
        if (ws) {
            console.log(`WebSocket readyState: ${ws.readyState}`);
            switch (ws.readyState) {
                case WebSocket.CONNECTING:
                    console.log("WebSocket is in CONNECTING state");
                    break;
                case WebSocket.OPEN:
                    console.log("WebSocket is in OPEN state");
                    break;
                case WebSocket.CLOSING:
                    console.log("WebSocket is in CLOSING state");
                    break;
                case WebSocket.CLOSED:
                    console.log("WebSocket is in CLOSED state");
                    break;
                default:
                    console.log("WebSocket is in unknown state");
            }
        }

        // 检查音频上下文状态
        if (audioContext) {
            console.log(`Audio context state during WebSocket error: ${audioContext.state}`);
        }

        // Error handling is managed by the close event handler
    });
}

// 存储WebRTC信令回调
type SignalingCallback = (data: any) => void;
type CallbackId = string;

// 使用Map存储每种类型的多个回调
const signalingCallbacks: Map<string, Map<CallbackId, SignalingCallback>> = new Map();

// 生成唯一的回调ID
function generateCallbackId(): CallbackId {
    return `callback_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// 注册信令回调，返回回调ID用于后续移除
export function registerSignalingCallback(type: string, callback: SignalingCallback): CallbackId {
    if (!signalingCallbacks.has(type)) {
        signalingCallbacks.set(type, new Map());
    }

    const callbackId = generateCallbackId();
    signalingCallbacks.get(type)!.set(callbackId, callback);
    return callbackId;
}

// 移除指定ID的信令回调
export function removeSignalingCallback(type: string, callbackId: CallbackId): void {
    if (signalingCallbacks.has(type)) {
        signalingCallbacks.get(type)!.delete(callbackId);
        // 如果该类型没有回调了，清理Map
        if (signalingCallbacks.get(type)!.size === 0) {
            signalingCallbacks.delete(type);
        }
    }
}

async function handleWebSocketMessage(event: MessageEvent): Promise<void> {
    try {
        // Check if the message is a text message
        if (typeof event.data === "string") {
            try {
                const jsonData = JSON.parse(event.data);
                if (jsonData.type === "pong") {
                    console.log("Received heartbeat pong from server");
                    return;
                }

                // 处理WebRTC信令消息
                if (jsonData.type === "answer" || jsonData.type === "ice_candidate_response") {
                    console.log(`Received WebRTC signaling message: ${jsonData.type}`);
                    const callbacksMap = signalingCallbacks.get(jsonData.type);

                    if (callbacksMap && callbacksMap.size > 0) {
                        // 调用所有注册的回调
                        callbacksMap.forEach((callback) => {
                            callback(jsonData);
                        });
                    } else {
                        // !NOTE: get one response, maybe ice_candidate_response register some callback, but no problem
                        //console.warn(`No callbacks registered for signaling message type: ${jsonData.type}`);
                    }
                    return;
                }
            } catch (e) {
                // Not JSON, continue processing as binary
            }
        }

        // 确保音频上下文处于运行状态
        if (audioContext && audioContext.state === 'suspended') {
            try {
                await audioContext.resume();
                console.log("Resumed audio context");
            } catch (err) {
                console.error("Error resuming audio context:", err);
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
    if (animationJsonStr == null || animationJsonStr.length == 0 || animationJsonStr == "[]") {
        // no animation, return 
        return true;
    }

    // 声明在函数范围内，以便在音频解码回调中使用
    let animationData = {};
    try {
        animationData = JSON.parse(animationJsonStr);
    } catch (error) {
        console.error("Error parsing animation JSON:", error);
        return false;
    }
    // 不立即更新动画数据，而是在音频播放时同步更新

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
    console.log("Audio length:", parsedFrame.animationAudio.audio.length);
    const audioVector = Array.from(parsedFrame.animationAudio.audio);
    const audioArray = new Uint8Array(audioVector);

    // 解码音频数据并将其添加到队列中，而不是直接播放
    audioContext.decodeAudioData(audioArray.buffer,
        function (buffer) {
            if (!audioContext) {
                console.warn("Audio context is not available after decoding");
                return;
            }

            console.log(`Decoded audio buffer, duration: ${buffer.duration.toFixed(3)}s`);

            // 将解码后的音频添加到队列中
            audioQueue.push({
                buffer: buffer,
                animationData: animationData,
                duration: buffer.duration
            });
            console.log(`Added audio to queue, queue length: ${audioQueue.length}`);

            // 如果队列处理器没有运行，启动它
            if (!isProcessingQueue) {
                console.log("Starting queue processor");
                processAudioQueue();
            }
        },
        function (err) {
            console.error("Error decoding audio data:", err);
        }
    );

    return true;
}

// 处理音频队列，确保音频片段按顺序播放，不重叠
function processAudioQueue(): void {
    if (!audioContext) {
        console.warn("Audio context is not available");
        isProcessingQueue = false;
        return;
    }

    if (audioQueue.length === 0) {
        console.log("Audio queue is empty, stopping queue processing");
        isProcessingQueue = false;
        return;
    }

    isProcessingQueue = true;

    // 确保playTime至少是当前时间
    if (playTime < audioContext.currentTime) {
        console.log(`Adjusting playTime from ${playTime} to ${audioContext.currentTime}`);
        playTime = audioContext.currentTime;
    }

    // 获取队列中的第一个音频片段
    const item = audioQueue.shift();
    if (item === undefined) {
        console.warn("Audio item is undefined after shift, stopping queue processing");
        isProcessingQueue = false;
        return;
    }
    console.log(`Processing audio item, duration: ${item.duration.toFixed(3)}s, queue length: ${audioQueue.length}`);

    try {
        // 创建音频源节点
        const source = new AudioBufferSourceNode(audioContext);
        source.buffer = item.buffer;

        // 创建一个增益节点来控制音量，实现淡入效果
        const gainNode = audioContext.createGain();

        // 设置初始音量为0
        gainNode.gain.setValueAtTime(0, playTime);

        // 在短时间内（例如50毫秒）将音量从0淡入到1，消除开始播放时的噪音
        gainNode.gain.linearRampToValueAtTime(1, playTime + 0.05);

        // 在音频结束时处理队列中的下一个项目
        source.onended = () => {
            console.log("Audio playback ended, processing next item");

            // 如果队列为空，设置音频播放状态为false
            if (audioQueue.length === 0 && avatarInstance) {
                console.log("Audio queue is empty, setting audio playing state to false");
                avatarInstance.setAudioPlayingState(false);
            }

            // 短暂延迟以确保不会有重叠
            setTimeout(() => {
                processAudioQueue();
            }, 10);
        };


        // 在音频开始播放时更新动画数据和播放状态
        if (avatarInstance) {
            // 设置音频播放状态为true
            avatarInstance.setAudioPlayingState(true);

            if (item.animationData) {
                // 使用音频上下文的时间作为动画的时间基准
                avatarInstance.updateAnimationData(item.animationData, playTime);
            }
        }

        // 连接音频节点：source -> gainNode -> destination
        source.connect(gainNode);
        gainNode.connect(audioContext.destination);

        // 开始播放音频
        const startTime = playTime;
        source.start(startTime);
        console.log(`Started audio playback at time ${startTime.toFixed(3)}`);

        // 更新下一个音频片段的播放时间
        playTime = playTime + item.duration + 0.05; // 添加50毫秒的间隔，防止重叠
        console.log(`Next audio will play at ${playTime.toFixed(3)}`);
    } catch (error) {
        console.error("Error processing audio queue item:", error);
        // 出错时继续处理队列中的下一个项目
        setTimeout(() => {
            processAudioQueue();
        }, 10);
    }
}

// 添加一个标志来跟踪用户是否主动断开连接
let userInitiatedDisconnect = false;

export function stopAudio(closeWebsocket: boolean, userInitiated: boolean = false): void {
    console.log(`Stopping audio playback, user initiated: ${userInitiated}`);
    playTime = 0;
    isPlaying = false;

    // 如果是用户主动断开，设置标志
    if (userInitiated) {
        userInitiatedDisconnect = true;
    }

    // 清空音频队列
    audioQueue = [];
    isProcessingQueue = false;

    // 设置音频播放状态为false
    if (avatarInstance) {
        avatarInstance.setAudioPlayingState(false);
    }

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
        console.log("Closing WebSocket connection");
        ws.close(1000); // Normal closure
        ws = null;
    }

    if (scriptProcessor) {
        scriptProcessor.disconnect();
    }
    if (source) {
        source.disconnect();
    }

    // 如果音频上下文存在，移除事件监听器并暂停它
    if (audioContext) {
        // 移除状态变化事件监听器
        audioContext.removeEventListener('statechange', handleAudioContextStateChange);

        if (audioContext.state === 'running') {
            console.log("Suspending audio context");
            audioContext.suspend().catch(err => {
                console.error("Error suspending audio context:", err);
            });
        }
    }
}