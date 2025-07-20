import { getWebSocket, sendWebSocketMessage, waitForWebSocketReady, registerSignalingCallback, removeSignalingCallback } from "./websocket";

let localChannel: RTCDataChannel | null = null;

const waitForIceGatheringComplete = async (
    pc: RTCPeerConnection,
    timeoutMs: number = 2000
): Promise<void> => {
    if (pc.iceGatheringState === "complete") return;
    console.log(
        "Waiting for ICE gathering to complete. Current state:",
        pc.iceGatheringState
    );
    return new Promise((resolve) => {
        let timeoutId: NodeJS.Timeout;
        const checkState = () => {
            console.log("icegatheringstatechange:", pc.iceGatheringState);
            if (pc.iceGatheringState === "complete") {
                cleanup();
                resolve();
            }
        };
        const onTimeout = () => {
            console.warn(`ICE gathering timed out after ${timeoutMs} ms.`);
            cleanup();
            resolve();
        };
        const cleanup = () => {
            pc.removeEventListener("icegatheringstatechange", checkState);
            clearTimeout(timeoutId);
        };
        pc.addEventListener("icegatheringstatechange", checkState);
        timeoutId = setTimeout(onTimeout, timeoutMs);
        // Checking the state again to avoid any eventual race condition
        checkState();
    });
};

export const createSmallWebRTCConnection = async (
    audioStream: MediaStream,
    signalingServerUrl: string,
    peerID: string,
    onConnected: () => void,
    onDisconnected: () => void,
    onTrack: (e: RTCTrackEvent) => void
): Promise<RTCPeerConnection> => {
    try {
        // 确保WebSocket连接已建立
        await waitForWebSocketReady();
        const audioTrack = audioStream.getAudioTracks()[0];
        console.log("audioTrack", audioTrack);

        const config: RTCConfiguration = {
            iceServers: [
                {
                    urls: ["stun:stun.l.google.com:19302"], // Google's public STUN server
                },
            ],
        };
        const pc = new RTCPeerConnection(config);

        // SmallWebRTCTransport expects to receive both transceivers
        // pc.addTransceiver(audioTrack, { direction: 'sendrecv' })
        // Add local stream to peer connection
        pc.addTrack(audioTrack);

        // Add peer connection event listeners
        addPeerConnectionEventListeners(peerID, signalingServerUrl, pc, onConnected, onDisconnected, onTrack);
        // await waitForIceGatheringComplete(pc);

        // Caller create data channel and listen event (open, close, message)
        localChannel = pc.createDataChannel("messaging-channel", { ordered: true });
        localChannel.binaryType = "arraybuffer";
        localChannel.addEventListener("open", () => {
            console.log("Local channel open!");
        });
        localChannel.addEventListener("close", () => {
            console.log("Local channel closed!");
        });
        localChannel.addEventListener("message", (event: MessageEvent) => {
            console.log(`Received Remote message : ${event.data}`);
        });

        // if don't post ice candidates, and waitForIceGatheringComplete, 
        // need Caller Create offer
        // https://developer.mozilla.org/en-US/docs/Web/API/RTCPeerConnection/setLocalDescription#providing_your_own_offer_or_answer
        // await pc.setLocalDescription(await pc.createOffer());
        // const offer = pc.localDescription;

        // @NOTE: https://developer.mozilla.org/en-US/docs/Web/API/RTCPeerConnection/setLocalDescription#implicit_descriptions
        await pc.setLocalDescription();
        const offer = pc.localDescription;

        if (!offer) {
            throw new Error("Failed to create offer");
        }

        // send offer to server
        const answer: RTCSessionDescriptionInit = await offer_answer(
            peerID,
            signalingServerUrl,
            offer
        );

        console.log("answer", answer);

        // Caller Set remote description
        await pc.setRemoteDescription(answer);
        return pc;
    } catch (e) {
        console.error(e);
        throw e;
    }
};

const offer_answer = async (
    peerID: string,
    signalingServerUrl: string,
    offer: RTCSessionDescriptionInit
): Promise<RTCSessionDescriptionInit> => {
    // 确保WebSocket连接已建立
    await waitForWebSocketReady();

    // 创建一个Promise，等待服务器的answer响应
    let timeoutId: NodeJS.Timeout;
    let callbackId: string;

    const answerPromise = new Promise<RTCSessionDescriptionInit>((resolve, reject) => {
        // 注册一个回调，处理服务器的answer响应
        callbackId = registerSignalingCallback("answer", (data) => {
            console.log(`Received answer via WebSocket for peer: ${peerID}`);
            // 收到answer后，移除回调并解析Promise
            removeSignalingCallback("answer", callbackId);
            clearTimeout(timeoutId);
            resolve(data);
        });

        // 设置超时
        timeoutId = setTimeout(() => {
            removeSignalingCallback("answer", callbackId);
            reject(new Error("Timeout waiting for answer"));
        }, 120000); // 120秒超时
    });

    // 通过WebSocket发送offer消息
    console.log(`Sending offer via WebSocket for peer: ${peerID}`);
    await sendWebSocketMessage({
        type: "offer",
        peer_id: peerID,
        sdp: offer.sdp,
        type_value: offer.type
    });

    // 等待并返回answer
    return await answerPromise;
};

const addPeerConnectionEventListeners = (
    peerID: string,
    signalingServerUrl: string,
    pc: RTCPeerConnection,
    onConnected: () => void,
    onDisconnected: () => void,
    onTrack: (e: RTCTrackEvent) => void
): void => {
    // remote stream track
    pc.ontrack = (e: RTCTrackEvent) => {
        console.log('Received remote stream:', e.streams[0]);
        onTrack(e);
    };

    pc.onicecandidate = async (event: RTCPeerConnectionIceEvent) => {
        if (event.candidate) {
            console.log("New ICE candidate:", event.candidate);
        } else {
            console.log("All ICE candidates have been sent.");
        }

        if (!event.candidate || !event.candidate.candidate) {
            return;
        }

        const iceCandidate = {
            peer_id: peerID,
            candidate_sdp: event.candidate.candidate,
            sdpMid: event.candidate.sdpMid,
            sdpMLineIndex: event.candidate.sdpMLineIndex,
            usernameFragment: event.candidate.usernameFragment
        };

        console.log('Posting ICE candidate: ', iceCandidate);
        await ice_candidate(peerID, signalingServerUrl, iceCandidate);
    };

    pc.onconnectionstatechange = async () => {
        console.log("onconnectionstatechange", pc?.connectionState);
        const connectionState = pc?.connectionState;
        if (connectionState === "connected") {
            onConnected();
        } else if (connectionState === "disconnected") {
            onDisconnected();
        }
    };

    pc.oniceconnectionstatechange = async () => {
        console.log("oniceconnectionstatechange", pc?.iceConnectionState);
    };

};

const ice_candidate = async (
    peerID: string,
    signalingServerUrl: string,
    iceCandidate: Record<string, unknown>,
): Promise<void> => {
    // 确保WebSocket连接已建立
    await waitForWebSocketReady();

    // 创建一个Promise，等待服务器的ice_candidate_response响应
    let timeoutId: NodeJS.Timeout;
    let callbackId: string;

    const iceResponsePromise = new Promise<void>((resolve, reject) => {
        // 注册一个回调，处理服务器的ice_candidate_response响应
        callbackId = registerSignalingCallback("ice_candidate_response", (data) => {
            console.log(`Received ICE candidate response via WebSocket for peer: ${peerID}`);
            // 收到响应后，移除回调并解析Promise
            removeSignalingCallback("ice_candidate_response", callbackId);
            clearTimeout(timeoutId);
            resolve();
        });

        // 设置超时
        timeoutId = setTimeout(() => {
            removeSignalingCallback("ice_candidate_response", callbackId);
            reject(new Error("Timeout waiting for ice_candidate_response"));
        }, 120000); // 120秒超时
    });

    // 通过WebSocket发送ice_candidate消息
    console.log(`Sending ICE candidate via WebSocket for peer: ${peerID}`);
    await sendWebSocketMessage({
        type: "ice_candidate",
        ...iceCandidate
    });

    // 等待响应
    await iceResponsePromise;
}