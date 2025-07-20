import asyncio
from abc import ABC, abstractmethod

import modal
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketState

from .modal_webrtc_peer import ModalWebRtcPeer


class ModalWebRtcSignalingServer:
    """
    WebRTC signaling server implementation that mediates connections between client peers
    and Modal-based WebRTC peers. Implement using the `app.cls` decorator.

    This server:
    - Provides a WebSocket endpoint (/ws/{peer_id}) for client connections
    - Spawns Modal-based peer instances for each client connection
    - Handles the WebRTC signaling process by relaying messages between clients and Modal peers
    - Manages the lifecycle of Modal peer instances

    To use this class:
    1. Create a subclass implementing get_modal_peer_class() to return your ModalWebRtcPeer implementation
    2. Optionally override initialize() for custom server setup
    3. Optionally add a frontend route to the `web_app` attribute
    """

    @modal.enter()
    def _initialize(self):
        self.web_app = FastAPI()

        # handle signaling through websocket endpoint
        @self.web_app.websocket("/ws/{peer_id}")
        async def ws(client_websocket: WebSocket, peer_id: str):
            await client_websocket.accept()
            await self._mediate_negotiation(client_websocket, peer_id)

        self.initialize()

    def initialize(self):
        pass

    @abstractmethod
    def get_modal_peer_class(self) -> type[ModalWebRtcPeer]:
        """
        Abstract method to return the `ModalWebRtcPeer` implementation to use.
        """
        raise NotImplementedError("Implement `get_modal_peer` to use `ModalWebRtcSignalingServer`")

    @modal.asgi_app()
    def web(self):
        return self.web_app

    async def _mediate_negotiation(self, websocket: WebSocket, peer_id: str):
        modal_peer_class = self.get_modal_peer_class()
        if not any(base.__name__ == "ModalWebRtcPeer" for base in modal_peer_class.__bases__):
            raise ValueError("Modal peer class must be an implementation of `ModalWebRtcPeer`")

        with modal.Queue.ephemeral() as q:
            print(f"Spawning modal peer instance for client peer {peer_id}...")
            modal_peer = modal_peer_class()
            modal_peer.run.spawn(q, peer_id)

            await asyncio.gather(
                relay_websocket_to_queue(websocket, q, peer_id),
                relay_queue_to_websocket(websocket, q, peer_id),
            )


async def relay_websocket_to_queue(websocket: WebSocket, q: modal.Queue, peer_id: str):
    while True:
        try:
            # get websocket message off queue and parse as json
            msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
            await q.put.aio(msg, partition=peer_id)

        except Exception:
            if WebSocketState.DISCONNECTED in [
                websocket.application_state,
                websocket.client_state,
            ]:
                return


async def relay_queue_to_websocket(websocket: WebSocket, q: modal.Queue, peer_id: str):
    while True:
        try:
            # get websocket message off queue and parse from json
            modal_peer_msg = await asyncio.wait_for(q.get.aio(partition="server"), timeout=0.5)

            if modal_peer_msg.startswith("close"):
                await websocket.close()
                return

            await websocket.send_text(modal_peer_msg)

        except Exception:
            if WebSocketState.DISCONNECTED in [
                websocket.application_state,
                websocket.client_state,
            ]:
                return
