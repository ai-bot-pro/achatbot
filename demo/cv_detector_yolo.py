from typing import Generator
import cv2
from ultralytics import YOLO
import supervision as sv


def yolo_detector(model_path: str = "yolov8n.pt"):
    # 加载选择的合适的预训练模型
    print("model-->:", model_path)
    model = YOLO(model_path)

    # Run predictions
    # results = model('https://ultralytics.com/images/bus.jpg')

    # 打开视频流（0表示摄像头，或使用视频文件路径）
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 进行流式目标检测
        results = model(frame, stream=True)
        print("results-->:", type(results))
        for item in results:
            print("item-->:", type(item), item)
            print("boxes-->:", item.boxes)
            detections = sv.Detections.from_ultralytics(item)
            print("detections-->:", type(detections), len(detections), detections)
            # 绘制检测结果
            annotated_frame = item.plot()
            print("annotated_frame-->:", type(annotated_frame))

            # 显示结果
            cv2.imshow('YOLOv8 Stream', annotated_frame)

        if detections.confidence.size > 0 and detections.confidence.max() > 0.5:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # NOTE: just test to print all models stream predict info :)
    yolo_detector("yolov8n.pt")
    yolo_detector("yolov10n.pt")
    yolo_detector("yolov8s-worldv2.pt")
    # yolo_detector("rtdetr-l.pt")  # v1
