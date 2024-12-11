import numpy as np
import os
import datetime
import cv2


def capture_img():
    cap = cv2.VideoCapture(0)
    i = 0
    while True:
        ret, frame = cap.read()
        k = cv2.waitKey(1)
        if k & 0xFF == 27:  # esc
            break
        elif k == ord("s"):
            cv2.imwrite("./images/cv_capture" + str(i) + ".jpg", frame)
            i += 1
        cv2.imshow("capture", frame)
    cap.release()
    cv2.destroyAllWindows()


# Standard Video Dimensions Sizes
"""
352 x 240 (240p) (SD) (VCD Players)
480 x 360 (360p)
858 x 480 (480p)
1280 x 720 (720p) (HD) (Some HDTVs)
1920 x 1080 (1080p) (HD) (Blu-Ray Players, HDTV)
3860 x 2160 (2160p) (Ultra-HD) (4K Players / Televisions)

Apart from these, there are other variants of resolutions as well
640 x 480 (VGA, Standard Definition TVs)
1280 x 544 (Wide-Screen movies)
1920 x 816 or 1920 x 800 (Wide-screen movies)
"""
STD_DIMENSIONS = {
    "240p": (352, 240),
    "360p": (480, 360),
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


def change_res(cap, width, height):
    """
    Set resolution for the video capture
    Function adapted from https://kirr.co/0l6qmh
    """
    cap.set(3, width)
    cap.set(4, height)


def get_dims(cap, res="1080p"):
    # grab resolution dimensions and set video capture to it.
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    # change the current caputre device
    # to the resulting resolution
    change_res(cap, width, height)
    return width, height


# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    ".avi": cv2.VideoWriter_fourcc(*"xvid"),
    # '.mp4': cv2.VideoWriter_fourcc(*'xvid'),
    # '.mp4': cv2.VideoWriter_fourcc(*'h264'),
    ".mp4": cv2.VideoWriter_fourcc(*"mp4v"),
}


def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE[".mp4"]


def capture_video():
    filename_dir = "./videos/cv"
    if not os.path.exists(filename_dir):
        os.makedirs(filename_dir)

    filename = (
        filename_dir
        + "/video_capture_"
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + ".mp4"
    )
    frames_per_second = 20.0
    res = "720p"
    res = "480p"

    cap = cv2.VideoCapture(0)
    out = cv2.VideoWriter(
        filename,
        get_video_type(filename),
        frames_per_second,
        get_dims(cap, res),
    )

    while True:
        ret, frame = cap.read()
        out.write(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def capture_videos():
    # 打开本地摄像头
    cap = cv2.VideoCapture(0)

    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    res = "480p"

    # Video Encoding, might require additional installs
    # Types of Codes: http://www.fourcc.org/codecs.php
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # 记录帧速率
    fps = 20.0

    # 记录每个视频的长度（以秒为单位）
    video_length_s = 3

    filename_dir = "./videos/cv"
    if not os.path.exists(filename_dir):
        os.makedirs(filename_dir)
    # 记录保存的文件名前缀
    filename_prefix = filename_dir + "/capture"

    # 记录已经写入的帧数
    frame_count = 0

    # 记录视频的序号
    video_number = 1

    # 持续捕获并保存视频帧
    while True:
        # 检查是否需要开始一个新的视频文件
        if frame_count == 0:
            # 创建输出视频文件
            filename = (
                filename_prefix
                + str(video_number)
                + "_"
                + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                + ".mp4"
            )
            out = cv2.VideoWriter(filename, fourcc, fps, get_dims(cap, res))
            print("Start recording " + filename)

        # 读取视频帧
        ret, frame = cap.read()

        # 检查是否到达视频长度的末尾
        if frame_count >= fps * video_length_s:
            # 释放资源
            out.release()
            cv2.destroyAllWindows()
            print("Finish recording " + filename)

            # 重置计数器和视频序号
            frame_count = 0
            video_number += 1
        else:
            # 处理帧
            # ...

            # 写入帧到输出视频文件
            out.write(frame)

            # 显示帧
            cv2.imshow("frame", frame)

            # 按 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # 更新计数器
            frame_count += 1

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


def list_file(file_suffix: str = ".mp4", data_dir="./videos/cv"):
    import glob

    filenames = sorted(glob.glob(os.path.join(data_dir, f"*{file_suffix}")))
    print(filenames)
    return filenames


if __name__ == "__main__":
    # capture_img()
    # capture_video()
    capture_videos()
    # list_file()
