import cv2
import os
import datetime
from flask import Flask, Response, request

app = Flask(__name__)

def generate_frames_from_video(video_path):
    """
    Generator function to read frames from a video and yield them as JPEG bytes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Encode the frame as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90] # JPEG quality 90
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame_bytes = buffer.tobytes()

        # Yield the frame in multipart/x-mixed-replace format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    print(f"Finished streaming frames from {video_path}")

@app.route('/stream_video_frames')
def stream_video_frames():
    """
    HTTP endpoint to stream video frames as a multipart/x-mixed-replace response.
    Usage: http://<server_ip>:<port>/stream_video_frames?video_path=<path_to_video.mp4>
    """
    video_path = request.args.get('video_path')

    if not video_path:
        return Response("Error: 'video_path' parameter is missing.", status=400)
    if not os.path.exists(video_path):
        return Response(f"Error: Video file not found at {video_path}", status=404)

    return Response(generate_frames_from_video(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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
    # Then open your browser to http://127.0.0.1:8001/stream_video_frames?video_path=videos/cv/capture4_2024-09-15_20-10-29.mp4
    # Or replace with an actual path to a video on your system.
    app.run(host='0.0.0.0', port=8001, debug=True)
