import cv2
import os
import datetime
import time
from flask import Flask, Response, request

app = Flask(__name__)


def generate_frames_from_video(video_source):
    """
    Generator function to read frames from a video source (path or URL) and yield them as JPEG bytes.
    The frames are yielded at a rate consistent with the original video's FPS.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{fps=},{total_frames=}")

    # Calculate the delay needed between frames to match the original video's FPS
    # If fps is 0 (e.g., for some image streams), avoid division by zero
    delay_s = 1.0 / fps if fps > 0 else 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Encode the frame as JPEG
        height, width, channel = frame.shape
        print(f"Frame dimensions: {width=}x{height=} {channel=}")
        # OpenCV captures frames in BGR format by default.
        # If the intention is to convert to RGB before encoding, perform the conversion.
        # This check assumes 3-channel images from video capture are BGR and need conversion to RGB.
        if len(frame.shape) == 3 and frame.shape[2] == 3:  # Check if it's a 3-channel image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # JPEG quality 90
        _, buffer = cv2.imencode(".jpg", frame, encode_param)
        frame_bytes = buffer.tobytes()

        # Yield the frame in multipart/x-mixed-replace format
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        # Introduce a delay to match the video's original FPS
        if delay_s > 0:
            time.sleep(delay_s)

    cap.release()
    print(f"Finished streaming frames from {video_source}")


@app.route("/stream_video_frames")
def stream_video_frames():
    """
    HTTP endpoint to stream video frames as a multipart/x-mixed-replace response.
    The 'video_source' parameter can be a local file path or an HTTP URL to a video file.
    Usage: http://<server_ip>:<port>/stream_video_frames?video_source=<path_or_url_to_video>
    """
    video_source = request.args.get("video_source")

    if not video_source:
        return Response("Error: 'video_source' parameter is missing.", status=400)

    # If it's a local file path, check if it exists
    if not video_source.startswith(("http://", "https://")) and not os.path.exists(video_source):
        return Response(f"Error: Local video file not found at {video_source}", status=404)

    return Response(
        generate_frames_from_video(video_source),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# The original video2Images function is kept for completeness if saving to disk is still needed
def video2Images(video_path: str, output_dir: str):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{fps=},{total_frames=}")

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取视频帧
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 生成输出文件名
        filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")

        # 保存帧
        cv2.imwrite(filename, frame)
        frame_count += 1

    # 释放资源
    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")


if __name__ == "__main__":
    # Example for saving images (original functionality)
    # video_path_save = "./videos/cv/capture4_2024-09-15_20-10-29.mp4"
    # output_dir_save = "./images/cv_saved"
    # if os.path.exists(video_path_save):
    #     video2Images(video_path_save, output_dir_save)
    # else:
    #     print(f"Video for saving not found: {video_path_save}. Skipping image extraction.")

    # Run the Flask app for streaming
    # To test, ensure 'videos/cv/capture4_2024-09-15_20-10-29.mp4' exists relative to this script.
    # Then open your browser to http://127.0.0.1:8001/stream_video_frames?video_source=videos/cv/capture4_2024-09-15_20-10-29.mp4
    # To stream from an HTTP URL (e.g., a public MP4 file):
    # http://127.0.0.1:8001/stream_video_frames?video_source=https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4
    app.run(host="0.0.0.0", port=8001, debug=True)
