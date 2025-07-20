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
    // Caller Send peer connect offer to server, get answer from server
    const serverUrl = new URL(`/api/offer/${peerID}`, signalingServerUrl.trim()).toString();
    console.log("Connecting to webrtc signaling server offer api:", serverUrl);
    const response = await fetch(serverUrl, {
        body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
        headers: { "Content-Type": "application/json" },
        method: "POST",
    });

    const answer = await response.json();
    return answer;
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
    const serverUrl = new URL(`/api/ice_candidate/${peerID}`, signalingServerUrl.trim()).toString();
    console.log("ice_candidate api:", serverUrl);
    const response = await fetch(serverUrl, {
        body: JSON.stringify(iceCandidate),
        headers: { "Content-Type": "application/json" },
        method: "POST",
    });
    console.log("ice_candidate response:", response);
}