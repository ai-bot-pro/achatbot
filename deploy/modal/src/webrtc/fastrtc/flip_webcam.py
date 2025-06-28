import modal

web_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]==0.115.4",
    "fastrtc==0.0.23",
    "gradio==5.7.1",
    "opencv-python-headless==4.11.0.86",
)

app = modal.App("fastrtc-flip-webcam", image=web_image)


TRACK_CONSTRAINTS = {
    "width": {"exact": 640},
    "height": {"exact": 480},
    "frameRate": {"min": 30},
    "facingMode": {  # https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackSettings/facingMode
        "ideal": "user"
    },
}

RTC_CONFIG = {"iceServers": [{"url": "stun:stun.l.google.com:19302"}]}


MAX_CONCURRENT_STREAMS = 10  # number of peers per instance on Modal

MINUTES = 60  # seconds
TIME_LIMIT = 10 * MINUTES  # time limit


"""
modal serve src/webrtc/fastrtc/flip_webcam.py 
"""


@app.function(
    max_containers=1,
    scaledown_window=TIME_LIMIT + 1 * MINUTES,  # add a small buffer to time limit
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_STREAMS)  # inputs per container
@modal.asgi_app()  # ASGI on Modal
def ui():
    import fastrtc  # WebRTC in Gradio
    import gradio as gr  # WebUIs in Python
    from fastapi import FastAPI  # asynchronous ASGI server framework
    from gradio.routes import mount_gradio_app  # connects Gradio and FastAPI
    from fastapi.responses import RedirectResponse

    app = FastAPI()


    @app.get("/ui/static/fonts/system-ui/system-ui-Regular.woff2")
    @app.get("/ui/static/fonts/ui-sans-serif/ui-sans-serif-Regular.woff2")
    @app.get("/favicon.ico")
    def get_font():
        # remove confusing error
        return {}

    with gr.Blocks() as blocks:  # block-wise UI definition
        gr.HTML(  # simple HTML header
            "<h1 style='text-align: center'>Streaming Video Processing with Modal and FastRTC</h1>"
        )

        with gr.Column():  # a column of UI elements
            fastrtc.Stream(  # high-level media streaming UI element
                modality="video",
                mode="send-receive",
                # handler=flip_vertically,  # handler -- handle incoming frame, produce outgoing frame
                handler=dummy,
                ui_args={"title": "Click 'Record' to flip your webcam in the cloud"},
                rtc_configuration=RTC_CONFIG,
                track_constraints=TRACK_CONSTRAINTS,
                concurrency_limit=MAX_CONCURRENT_STREAMS,  # limit simultaneous connections
                time_limit=TIME_LIMIT,  # limit time per connection
            )

    return mount_gradio_app(app=app, blocks=blocks, path="/")


def dummy(image):
    print(f"{image=}")
    return image


def flip_vertically(image):
    import cv2
    import numpy as np

    image = image.astype(np.uint8)

    if image is None:
        print("failed to decode image")
        return

    # flip vertically and caption to show video was processed on Modal
    image = cv2.flip(image, 0)
    lines = ["Hello from Modal!"]
    caption_image(image, lines)

    return image


def caption_image(img, lines, font_scale=0.8, thickness=2, margin=10, font=None, color=None):
    import cv2

    if font is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
    if color is None:
        color = (127, 238, 100, 128)  # Modal Green

    # get text sizes
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    if not sizes:
        return

    # position text in bottom right
    pos_xs = [img.shape[1] - size[0] - margin for size in sizes]

    pos_ys = [img.shape[0] - margin]
    for _width, height in reversed(sizes[:-1]):
        next_pos = pos_ys[-1] - 2 * height
        pos_ys.append(next_pos)

    for line, pos in zip(lines, zip(pos_xs, reversed(pos_ys))):
        cv2.putText(img, line, pos, font, font_scale, color, thickness)
