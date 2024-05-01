from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO

# THIN_BBOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
annotator = sv.BoxAnnotator(thickness=2, text_padding=5)

# def add_annotations_to_image(img, is_thick, detections):
#     annotator = THICK_BBOX_ANNOTATOR if is_thick else THIN_BBOX_ANNOTATOR
#     return annotator.annotate(scene=img.copy(), detections=detections)

model = YOLO("runs/detect/exdark640key/train-16b-32e/weights/best.pt")
dataset_pth = Path("datasets/exdark/yolo/exdark640testonly")
dataset = sv.DetectionDataset.from_yolo(images_directory_path=dataset_pth / "images",
                                        annotations_directory_path=dataset_pth / "labels",
                                        data_yaml_path=dataset_pth / "data.yaml")

LABELS = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table', 'Key']

def add_annotations_to_image(img, detections):
    return annotator.annotate(scene=img.copy(), detections=detections, labels=['People'])


for name, detection in dataset.annotations.items():
    results = model.predict(name, save=False, conf=0.5)
    for r in results:
        print(r.boxes.xywhn.tolist())
        print(r.boxes.conf.tolist())
        print(r.boxes.cls.tolist())
    # predicted_pth = f"runs/detect/predict/{Path(name).name}"
    # img = cv2.imread(predicted_pth)
    # img = add_annotations_to_image(img, detection)
    # cv2.imwrite(filename=predicted_pth, img=img)
    break
