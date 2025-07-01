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
    addPeerConnectionEventListeners(pc, onConnected, onDisconnected, onTrack);
    await waitForIceGatheringComplete(pc);

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

    // Caller Create offer
    await pc.setLocalDescription(await pc.createOffer());
    const offer = pc.localDescription;

    if (!offer) {
        throw new Error("Failed to create offer");
    }

    // Caller Send peer connect offer to server, get answer from server
    let serverUrl = signalingServerUrl.trim();
    if (!serverUrl) {
        serverUrl = "/api/offer";
    } else if (
        !serverUrl.startsWith("http://") &&
        !serverUrl.startsWith("https://") &&
        !serverUrl.startsWith("/")
    ) {
        serverUrl = "/" + serverUrl;
    }
    serverUrl = serverUrl + "/" + peerID;
    console.log("Connecting to webrtc signaling server:", serverUrl);
    const response = await fetch(serverUrl, {
        body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
        headers: { "Content-Type": "application/json" },
        method: "POST",
    });
    const answer: RTCSessionDescriptionInit = await response.json();

    // Caller Set remote description
    await pc.setRemoteDescription(answer);
    return pc;
};

const addPeerConnectionEventListeners = (
    pc: RTCPeerConnection,
    onConnected: () => void,
    onDisconnected: () => void,
    onTrack: (e: RTCTrackEvent) => void
): void => {
    // remote stream track
    pc.ontrack = (e: RTCTrackEvent) => {
        onTrack(e);
    };

    pc.oniceconnectionstatechange = () => {
        console.log("oniceconnectionstatechange", pc?.iceConnectionState);
    };

    pc.onconnectionstatechange = () => {
        console.log("onconnectionstatechange", pc?.connectionState);
        const connectionState = pc?.connectionState;
        if (connectionState === "connected") {
            onConnected();
        } else if (connectionState === "disconnected") {
            onDisconnected();
        }
    };

    pc.onicecandidate = (event: RTCPeerConnectionIceEvent) => {
        if (event.candidate) {
            console.log("New ICE candidate:", event.candidate);
        } else {
            console.log("All ICE candidates have been sent.");
        }
    };
};