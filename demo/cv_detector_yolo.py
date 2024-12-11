import cv2
from ultralytics import YOLO
import supervision as sv


def yolo_detector(model_path: str = "yolov8n.pt"):
    # 加载选择的合适的预训练模型
    model = YOLO(model_path)
    print("model-->:", model_path, type(model), model)
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"have {model_million_params}M parameters")
    print("device-->:", model.device)

    # Define custom classes
    if hasattr(model, "set_classes"):
        model.set_classes(["person"])
        print(f"{model_path} set classes")
    # Run predictions
    # results = model('https://ultralytics.com/images/bus.jpg')

    # 打开视频流（0表示摄像头，或使用视频文件路径）
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        print("cv read frame", type(frame))

        # 进行流式目标检测
        results = model(frame, stream=True)
        print("results-->:", type(results))
        is_check = False
        for item in results:
            print("item-->:", type(item), item)
            print("boxes-->:", item.boxes)
            detections = sv.Detections.from_ultralytics(item)
            print(
                "detections-->:",
                type(detections),
                len(detections),
                detections,
                detections.data["class_name"],
            )
            # 绘制检测结果
            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence in zip(detections["class_name"], detections.confidence)
            ]

            BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
            LABEL_ANNOTATOR = sv.LabelAnnotator(
                text_thickness=2, text_scale=1, text_color=sv.Color.BLACK
            )
            annotated_image = frame.copy()
            annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
            annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections, labels=labels)
            annotated_frame = annotated_image
            # annotated_frame = item.plot()

            print("annotated_frame-->:", type(annotated_frame))

            # 显示结果
            cv2.imshow("YOLOv8 Stream", annotated_frame)
            cf_dict = {}
            for name in detections.data["class_name"]:
                cf_dict[name] = {"d_cn": 0, "d_cf": []}
            for t in zip(
                detections.xyxy,
                detections.confidence,
                detections.class_id,
                detections.data["class_name"],
            ):
                cf_dict[t[3]]["d_cn"] += 1
                cf_dict[t[3]]["d_cf"].append(t[1])

            if (
                "person" in cf_dict
                and max(cf_dict["person"]["d_cf"]) > 0.5
                and cf_dict["person"]["d_cn"] >= 1
            ):
                print("person detection-->:", cf_dict)
                is_check = True

        if is_check:
            # pass
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # NOTE: just test to print all models stream predict info :)
    # yolo_detector("./models/yolov8n.pt")
    # yolo_detector("./models/yolov10n.pt")
    yolo_detector("./models/yolo11n.pt")
    # yolo_detector("./models/yolov8s-worldv2.pt")
    # yolo_detector("./models/rtdetr-l.pt")  # v1
