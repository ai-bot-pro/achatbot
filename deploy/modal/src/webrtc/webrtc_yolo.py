import os
from pathlib import Path

import modal

from .modal_webrtc import ModalWebRtcPeer, ModalWebRtcSignalingServer

py_version = "3.12"
tensorrt_ld_path = f"/usr/local/lib/python{py_version}/site-packages/tensorrt_libs"

video_processing_image = (
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
)

CACHE_VOLUME = modal.Volume.from_name("webrtc-yolo-cache", create_if_missing=True)
CACHE_PATH = Path("/cache")
cache = {CACHE_PATH: CACHE_VOLUME}

app = modal.App("example-webrtc-yolo")


@app.cls(
    image=video_processing_image,
    gpu="L4",
    volumes=cache,
    secrets=[modal.Secret.from_name("turn-credentials")],
    # cloud="aws",
    # region="ap-northeast"
    min_containers=1,
    max_containers=2,  # when peers per GPU container reach max input, scale up, max 2
)
@modal.concurrent(
    target_inputs=2,  # try to stick to just two peers per GPU container
    max_inputs=3,  # but allow up to three
)
class ObjDet(ModalWebRtcPeer):
    async def initialize(self):
        self.yolo_model = get_yolo_model(CACHE_PATH)

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
        def on_track(track: MediaStreamTrack) -> None:
            print(f"Video Processor, {self.id}, received {track.kind} track {track} from {peer_id}")

            if track.kind == "video":
                output_track = get_yolo_track(track, self.yolo_model)  # see Addenda
                self.pcs[peer_id].addTrack(output_track)
            elif track.kind == "audio":
                # For audio track, we just echo it back
                self.pcs[peer_id].addTrack(track)

            # keep us notified when the incoming track ends
            @track.on("ended")
            async def on_ended() -> None:
                print(
                    f"Video Processor, {self.id}, incoming {track.kind} track from {peer_id} ended"
                )

    async def get_turn_servers(self, peer_id=None, msg=None) -> dict:
        print(f"get_turn_servers called for {peer_id} {msg}")
        try:
            turn_servers = await get_cloudflare_turn_servers()
            # turn_servers = await get_metered_turn_servers()
        except Exception as e:
            print(e)
            return {"type": "error", "message": str(e)}

        return {"type": "turn_servers", "ice_servers": turn_servers}


# ### Implement a `SignalingServer`

# The `ModalWebRtcSignalingServer` class is much simpler to implement.
# The main thing we need to do is implement the `get_modal_peer_class` method which will return our implementation of the `ModalWebRtcPeer` class, `ObjDet`.
#
# It also has an `initialize()` method we can optionally override (called at the beginning of the [container lifecycle](https://modal.com/docs/guides/lifecycle-functions))
# as well as a `web_app` property which will be [served by Modal](https://modal.com/docs/guide/webhooks#asgi-apps---fastapi-fasthtml-starlette).
# We'll use these to add a frontend which uses the WebRTC JavaScript API to stream a peer's webcam from the browser.
#
# The JavaScript and HTML files are alongside this example in the [Github
# repo](https://github.com/modal-labs/modal-examples/tree/main/07_web_endpoints/webrtc/frontend).

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
class WebcamObjDet(ModalWebRtcSignalingServer):
    def get_modal_peer_class(self):
        return ObjDet

    def initialize(self):
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles

        self.web_app.mount("/static", StaticFiles(directory="/frontend"))

        @self.web_app.get("/")
        async def root():
            html = open("/frontend/index.html").read()
            return HTMLResponse(content=html)

        print("----initialized singaling server----")


def get_yolo_model(cache_path):
    import onnxruntime

    from .yolo import YOLOv10

    onnxruntime.preload_dlls()
    return YOLOv10(cache_path)


def get_yolo_track(track, yolo_model=None):
    import numpy as np
    import onnxruntime
    from aiortc import MediaStreamTrack
    from aiortc.contrib.media import VideoFrame

    from .yolo import YOLOv10

    class YOLOTrack(MediaStreamTrack):
        """
        Custom media stream track performs object detection
        on the video stream and passes it back to the source peer
        """

        kind: str = "video"
        conf_threshold: float = 0.15

        def __init__(self, track: MediaStreamTrack, yolo_model=None) -> None:
            super().__init__()

            self.track = track
            if yolo_model is None:
                onnxruntime.preload_dlls()
                self.yolo_model = YOLOv10(CACHE_PATH)
            else:
                self.yolo_model = yolo_model

        def detection(self, image: np.ndarray) -> np.ndarray:
            import cv2

            orig_shape = image.shape[:-1]

            image = cv2.resize(
                image,
                (self.yolo_model.input_width, self.yolo_model.input_height),
            )

            image = self.yolo_model.detect_objects(image, self.conf_threshold)

            image = cv2.resize(image, (orig_shape[1], orig_shape[0]))

            return image

        # this is the essential method we need to implement
        # to create a custom MediaStreamTrack
        async def recv(self) -> VideoFrame:
            frame = await self.track.recv()
            img = frame.to_ndarray(format="bgr24")

            processed_img = self.detection(img)

            # VideoFrames are from a really nice package called av
            # which is a pythonic wrapper around ffmpeg
            # and a dependency of aiortc
            new_frame = VideoFrame.from_ndarray(processed_img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base

            return new_frame

    return YOLOTrack(track)


async def get_cloudflare_turn_servers(ttl=86400):
    import aiohttp

    auth_token = os.environ["CLOUDFLARE_TURN_API_TOKEN"]
    key_id = os.environ["CLOUDFLARE_TURN_TOKEN"]
    url = f"https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate-ice-servers"

    print(url)
    headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}

    data = {"ttl": ttl}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status not in [200, 201]:
                error_text = await response.text()
                raise Exception(f"error status {response.status} {error_text}")

            data = await response.json()
            return data["iceServers"]


async def get_metered_turn_servers():
    import aiohttp

    turn_name = os.environ.get("METERED_TURN_USERNAME")
    api_key = os.environ.get("METERED_TURN_API_KEY")
    url = f"https://{turn_name}/api/v1/turn/credentials?apiKey={api_key}"
    print(url)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status not in [200, 201]:
                error_text = await response.text()
                raise Exception(f"error status {response.status} {error_text}")

            data = await response.json()
            return data
