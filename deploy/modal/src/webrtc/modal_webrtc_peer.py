import os
import asyncio
import json
from abc import ABC, abstractmethod

import modal

from .turn_server import get_metered_turn_servers, get_cloudflare_turn_servers


class ModalWebRtcPeer(ABC):
    """
    Base class for implementing WebRTC peer connections in Modal using aiortc.
    Implement using the `app.cls` decorator.

    This class provides a complete WebRTC peer implementation that handles:
    - Peer connection lifecycle management (creation, negotiation, cleanup)
    - Signaling via Modal Queue for SDP offer/answer exchange and ICE candidate handling
    - Automatic STUN server configuration (defaults to Google's STUN server)
    - Stream setup and management

    Required methods to override:
    - setup_streams(): Implementation for setting up media tracks and streams

    Optional methods to override:
    - initialize(): Custom initialization logic when peer is created
    - run_streams(): Implementation for stream runtime logic
    - get_turn_servers(): Implementation to provide custom TURN server configuration
    - exit(): Custom cleanup logic when peer is shutting down

    The peer connection is established through a ModalWebRtcSignalingServer that manages
    the signaling process between this peer and client peers.
    """

    @modal.enter()
    async def _initialize(self):
        import shortuuid
        from aiortc import RTCPeerConnection

        self.id = shortuuid.uuid()
        self.pcs: dict[str, RTCPeerConnection] = {}
        self.pending_candidates = {}
        self.turn_server = os.getenv("TURN_SERVER", "cloudflare")

        # call custom init logic
        await self.initialize()

    async def initialize(self):
        """Override to add custom logic when creating a peer"""

    @abstractmethod
    async def setup_streams(self, peer_id):
        """Override to add custom logic when creating a connection and setting up streams"""
        raise NotImplementedError

    async def run_streams(self, peer_id):
        """Override to add custom logic when running streams"""

    async def get_turn_servers(self, peer_id=None, msg=None) -> dict:
        print(f"get_turn_servers called for {peer_id} {msg}")
        try:
            if self.turn_server == "metered":
                turn_servers = await get_metered_turn_servers()
            else:
                turn_servers = await get_cloudflare_turn_servers()
        except Exception as e:
            return {"type": "error", "message": str(e)}

        return {"type": "turn_servers", "ice_servers": turn_servers}

    async def _setup_peer_connection(self, peer_id):
        """Creates an RTC peer connection via an ICE server"""
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection

        # aiortc automatically uses google's STUN server,
        # but we can also specify our own
        config = RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")])
        self.pcs[peer_id] = RTCPeerConnection(configuration=config)
        self.pending_candidates[peer_id] = []
        await self.setup_streams(peer_id)

        print(f"Created peer connection and setup streams from {self.id} to {peer_id}")

    @modal.method()
    async def run(self, queue: modal.Queue, peer_id: str):
        """Run the RTC peer after establishing a connection by passing WebSocket messages over a Queue."""
        print(f"Running modal peer instance for client peer {peer_id}...")

        await self._connect_over_queue(queue, peer_id)
        await self._run_streams(peer_id)

    async def _connect_over_queue(self, queue: modal.Queue, peer_id: str):
        """Connect this peer to another by passing messages along a Modal Queue."""

        msg_handlers = {  # message types we need to handle
            "offer": self.handle_offer,  # SDP offer
            "ice_candidate": self.handle_ice_candidate,  # trickled ICE candidate
            "identify": self.get_identity,  # identify challenge
            "get_turn_servers": self.get_turn_servers,  # TURN server request
        }

        while True:
            try:
                if self.pcs.get(peer_id) and (
                    self.pcs[peer_id].connectionState in ["connected", "closed", "failed"]
                ):
                    await queue.put.aio("close", partition="server")
                    break

                # read and parse websocket message passed over queue
                msg = json.loads(
                    await asyncio.wait_for(queue.get.aio(partition=peer_id), timeout=0.5)
                )
                # print(f"{msg=}")

                # dispatch the message to its handler
                if handler := msg_handlers.get(msg.get("type")):
                    response = await handler(peer_id, msg)
                else:
                    print(f"Unknown message type: {msg.get('type')}")
                    response = None

                # pass the message back over the queue to the server
                if response is not None:
                    # print(f"{response=}")
                    await queue.put.aio(json.dumps(response), partition="server")

            except Exception:
                continue

    async def _run_streams(self, peer_id):
        """Run WebRTC streaming with a peer."""
        print(f"Modal peer {self.id} running streams for {peer_id}...")

        await self.run_streams(peer_id)

        # run until connection is closed or broken
        while self.pcs[peer_id].connectionState == "connected":
            await asyncio.sleep(0.1)

        print(f"Modal peer {self.id} ending streaming for {peer_id}")

    async def handle_offer(self, peer_id, msg):
        """Handles a peers SDP offer message by producing an SDP answer."""
        from aiortc import RTCSessionDescription

        print(f"Peer {self.id} handling SDP offer from {peer_id}...")

        await self._setup_peer_connection(peer_id)
        await self.pcs[peer_id].setRemoteDescription(RTCSessionDescription(msg["sdp"], msg["type"]))

        answer = await self.pcs[peer_id].createAnswer()
        await self.pcs[peer_id].setLocalDescription(answer)
        sdp = self.pcs[peer_id].localDescription.sdp

        return {"sdp": sdp, "type": answer.type, "peer_id": self.id}

    async def handle_ice_candidate(self, peer_id, msg):
        """Add an ICE candidate sent by a peer."""
        from aiortc import RTCIceCandidate
        from aiortc.sdp import candidate_from_sdp

        candidate = msg.get("candidate")

        if not candidate:
            raise ValueError

        print(
            f"Modal peer {self.id} received ice candidate from {peer_id}: {candidate['candidate_sdp']}..."
        )

        # parse ice candidate
        ice_candidate: RTCIceCandidate = candidate_from_sdp(candidate["candidate_sdp"])
        ice_candidate.sdpMid = candidate["sdpMid"]
        ice_candidate.sdpMLineIndex = candidate["sdpMLineIndex"]

        if not self.pcs.get(peer_id):
            self.pending_candidates[peer_id].append(ice_candidate)
        else:
            if len(self.pending_candidates[peer_id]) > 0:
                [
                    await self.pcs[peer_id].addIceCandidate(c)
                    for c in self.pending_candidates[peer_id]
                ]
                self.pending_candidates[peer_id] = []
            await self.pcs[peer_id].addIceCandidate(ice_candidate)

    async def get_identity(self, peer_id=None, msg=None):
        """Reply to an identify message with own id."""
        return {"type": "identify", "peer_id": self.id}

    async def generate_offer(self, peer_id):
        print(f"Peer {self.id} generating offer for {peer_id}...")

        await self._setup_peer_connection(peer_id)
        offer = await self.pcs[peer_id].createOffer()
        await self.pcs[peer_id].setLocalDescription(offer)
        sdp = self.pcs[peer_id].localDescription.sdp

        return {"sdp": sdp, "type": offer.type, "peer_id": self.id}

    async def handle_answer(self, peer_id, answer):
        from aiortc import RTCSessionDescription

        print(f"Peer {self.id} handling answer from {peer_id}...")
        # set remote peer description
        await self.pcs[peer_id].setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )

    @modal.exit()
    async def _exit(self):
        print(f"Shutting down peer: {self.id}...")
        await self.exit()

        if self.pcs:
            print(f"Closing peer connections for peer {self.id}...")
            await asyncio.gather(*[pc.close() for pc in self.pcs.values()])
            self.pcs = {}

    async def exit(self):
        """Override with any custom logic when shutting down container."""
