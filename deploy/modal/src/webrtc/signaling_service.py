import os
from pathlib import Path

import modal

from .modal_webrtc_signaling_server import ModalWebRtcSignalingServer
from .bot_webrtc_peer import BotWebRtcPeer, app


base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("python3-opencv", "ffmpeg")
    .pip_install(
        "fastapi[standard]==0.115.4",
        "aiortc==1.11.0",
        "opencv-python==4.11.0.86",
        "shortuuid==1.0.13",
    )
)

this_directory = Path(__file__).parent.resolve()

server_image = base_image.add_local_dir(this_directory / "frontend", remote_path="/frontend")


@app.cls(
    image=server_image,
    max_containers=1,
)
class WebcamWebRtcSignalingServer(ModalWebRtcSignalingServer):
    def get_modal_peer_class(self):
        return BotWebRtcPeer

    def initialize(self):
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles

        self.web_app.mount("/static", StaticFiles(directory="/frontend"))

        @self.web_app.get("/")
        async def root():
            html = open("/frontend/index.html").read()
            return HTMLResponse(content=html)

        print("----initialized singaling service----")
