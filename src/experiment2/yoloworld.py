import cv2 
import supervision as sv

import numpy as np
from tqdm import tqdm 
from inference.models.yolo_world.yolo_world import YOLOWorld

from pathlib import Path

if __name__ == "__main__":
    model = YOLOWorld(model_id="yolo_world/l")

    classes = ["Boat", "Bicycle", "Chair", "Bus", "Cat", "Car", "Table", "Bottle", "Dog", "Motorbike", "People", "Cup", "Key"]
    model.set_classes(classes)

    test_imgs_pth = Path("/home/xkostrna/night-detector/datasets/exdark/yolo/exdark640testonly/images")

    # 2015_00401.jpg

    padding = 32
    for i, img_pth in enumerate(test_imgs_pth.iterdir()):

        image = cv2.imread(str(img_pth))
        results = model.infer(image, confidence=0.003)

        detections = sv.Detections.from_inference(results).with_nms(threshold=0.1)
        BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
        LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=0.5, text_color=sv.Color.BLACK)

        labels = [
                f"{classes[class_id]} {confidence:.2f}"
                for class_id, confidence
                in zip(detections.class_id, detections.confidence)
        ]

        height, width = image.shape[:2]
        annotated_image = np.zeros((height + 2 * padding, width + 2 * padding, 3), dtype=np.uint8)
        annotated_image[padding:padding+height, padding:padding+width] = image.copy()
        # annotated_image = image.copy()
        detections.xyxy += padding
        annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
        annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections, labels)
        cv2.imwrite(f"yoloworld_inference/result_{i}.jpg", annotated_image)

        if i == 500:
            break
