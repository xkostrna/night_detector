import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("runs/detect/exdark640key/train-16b-32e/weights/best.pt")
byte_tracker = sv.ByteTrack()
annotator = sv.BoxAnnotator()


# https://roboflow.com/how-to-track/yolo-world
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame, conf=0.5)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = byte_tracker.update_with_detections(detections)
    labels = [f"{model.model.names[class_id]} {confidence:0.2f}"
              for _, _, confidence, class_id, tracker_id
              in detections]
    return annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)


sv.process_video(source_path="nyc_night.mp4",
                 target_path=f"result.mp4",
                 callback=callback)
