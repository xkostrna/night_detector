__authors__ = "Bc. Tomáš Koštrna, Ing. Pavol Marák, PhD."

from ultralytics import YOLO
import cv2


model = YOLO("yolov8n.pt")
DEFAULT_DATA = "coco128.yaml"


def track_video(src: str):
    global model
    model.track(source="https://youtu.be/Zgi9g1ksQHc",
                conf=0.3,
                iou=0.5,
                show=True)


def train_model_on_images(data: str, test_img: str):
    global model
    model.train(data=data,
                epochs=16,
                batch=32,
                workers=1,
                lr0=0.005,
                lrf=0.01,
                amp=False,
                device=0,
                imgsz=416)
    metrics = model.val()
    results = model(test_img)
    res_plotted = results[0].plot()
    cv2.imshow("result", res_plotted)
    cv2.waitKey()
    path = model.export(format="onnx")
    print(f"Metrics : {metrics}")
    print(f"Model exported to : {path}")


if __name__ == "__main__":
    train_model_on_images(data="F:/School/Ing/DIPLOMA/night_detector/datasets/exdark-yolo/exdark-yolo-all/data.yaml",
                          test_img="https://ultralytics.com/images/bus.jpg")
