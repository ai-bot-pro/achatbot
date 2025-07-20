import os
from pathlib import Path

import modal

from .modal_webrtc_peer import ModalWebRtcPeer

app = modal.App("demo-webrtc-bot")

py_version = "3.12"
tensorrt_ld_path = f"/usr/local/lib/python{py_version}/site-packages/tensorrt_libs"

video_audio_processing_image = (
    modal.Image.debian_slim(python_version=py_version)  # matching ld path
    # update locale as required by onnx
    .apt_install("locales")
    .run_commands(
        "sed -i '/^#\\s*en_US.UTF-8 UTF-8/ s/^#//' /etc/locale.gen",  # use sed to uncomment
        "locale-gen en_US.UTF-8",  # set locale
        "update-locale LANG=en_US.UTF-8",
    )
    .env({"LD_LIBRARY_PATH": tensorrt_ld_path, "LANG": "en_US.UTF-8"})
    # install system dependencies
    .apt_install("python3-opencv", "ffmpeg")
    # install Python dependencies
    .pip_install(
        "aiortc==1.11.0",
        "fastapi==0.115.12",
        "huggingface-hub[hf_xet]==0.30.2",
        "onnxruntime-gpu==1.21.0",
        "opencv-python==4.11.0.86",
        "tensorrt==10.9.0.34",
        "torch==2.7.0",
        "shortuuid==1.0.13",
    )
    .env(
        {
            "BOT_NAME": os.getenv("BOT_NAME", "echo"),
            "TURN_SERVER": os.getenv("TURN_SERVER", "cloudflare"),
        }
    )
)

CACHE_VOLUME = modal.Volume.from_name("webrtc-yolo-cache", create_if_missing=True)
CACHE_PATH = Path("/cache")
cache = {CACHE_PATH: CACHE_VOLUME}


@app.cls(
    image=video_audio_processing_image,
    gpu=os.getenv("IMAGE_GPU", None),
    volumes=cache,
    secrets=[modal.Secret.from_name("turn-credentials")],
    # u can use the nearest cloud region to run server
    # cloud="aws",
    # region="ap-northeast"
    min_containers=1,
    max_containers=2,  # when peers per GPU/CPU container reach max inputs, scale up, max 2
)
@modal.concurrent(
    target_inputs=2,  # try to stick to just two peers per GPU/CPU container
    max_inputs=3,  # but allow up to three
)
class BotWebRtcPeer(ModalWebRtcPeer):
    async def initialize(self):
        print(f"initialize ModalWebRtcPeer")
        # TODO: achatbot factory
        load()

    async def setup_streams(self, peer_id: str):
        from aiortc import MediaStreamTrack

        # keep us notified on connection state changes
        @self.pcs[peer_id].on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if self.pcs[peer_id]:
                print(
                    f"Video Processor, {self.id}, connection state to {peer_id}: {self.pcs[peer_id].connectionState}"
                )

        # when we receive a track from the source peer
        # we create a processed track and add it to our stream
        # back to the source peer
        @self.pcs[peer_id].on("track")
        def OnTrack(in_track: MediaStreamTrack) -> None:
            print(
                f"[OnTrack] {self.id}, received {in_track.kind} in_track({type(in_track)}) {in_track.__dict__} from {peer_id}"
            )

            # NOTE: track all kind include video, audio, bot track obj to recieve and process
            output_track = get_out_track(in_track)
            self.pcs[peer_id].addTrack(output_track)

            # keep us notified when the incoming track ends
            @in_track.on("ended")
            async def OnEnded() -> None:
                print(f"[OnEnded], {self.id}, incoming {in_track.kind} track from {peer_id} ended")


def load():
    bot_name = os.getenv("BOT_NAME", "echo")
    if bot_name == "detector_yolo":
        from .track.yolo import load

        load(CACHE_PATH)


def get_out_track(in_track):
    bot_name = os.getenv("BOT_NAME", "echo")
    if bot_name == "detector_yolo":
        from .track.yolo import YOLOTrack

        return YOLOTrack(in_track)
    else:
        from .track.base import BaseTrack

        return BaseTrack(in_track)
