__authors__ = "Bc. Tomáš Koštrna, Ing. Pavol Marák, PhD."

from pathlib import Path
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


def test_model(dataset_pth: Path, num_img: int):
    for i, img_pth in enumerate(dataset_pth.iterdir()):
        if i == num_img:
            break
        results = model(img_pth)
        res_plotted = results[0].plot()
        cv2.imshow("result", res_plotted)
        cv2.waitKey()


def train_model_on_images(data: Path, name: str = ''):
    global model
    model.train(data=data,
                epochs=50,
                batch=32,
                nbs=32,
                workers=1,
                amp=False,
                device=0,
                # name=name,
                imgsz=416)
    metrics = model.val()
    path = model.export(format="onnx")
    print(f"Metrics : {metrics}")
    print(f"Model exported to : {path}")


def main():
    data_pth = Path("F:/School/Ing/DIPLOMA/night_detector/datasets/exdark-yolo/exdark-yolo-green/data.yaml")
    train_model_on_images(data=data_pth)


if __name__ == "__main__":
    main()
