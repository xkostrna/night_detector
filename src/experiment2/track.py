# implemented using: https://roboflow.com/how-to-track/yolov8
import time
import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("/home/xkostrna/night-detector/runs/detect/exdark640key/train-16b-32e/weights/best.pt")
byte_tracker = sv.ByteTrack()
annotator = sv.BoxAnnotator()
total_frames = 0


def callback1(frame:np.ndarray) -> sv.Detections:
    result = model(frame, conf=0.5, verbose=False)[0]
    return sv.Detections.from_ultralytics(result)


slicer = sv.InferenceSlicer(callback=callback1)


def callback2(frame: np.ndarray, index: int) -> np.ndarray:
    global total_frames
    results = model(frame, conf=0.5, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    # detections = slicer(image=frame)
    detections = byte_tracker.update_with_detections(detections)
    labels = [f"{model.model.names[class_id]} {confidence:0.2f}"
              for _, _, confidence, class_id, tracker_id, _
              in detections]
    total_frames += 1
    return annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)


start_time = time.time()
sv.process_video(source_path="/home/xkostrna/night-detector/videos/test_video_android.mp4",
                 target_path="android_labeled_sliced.mp4",
                 callback=callback2)
elapsed_time = time.time() - start_time

print(f"Total frames processed: {total_frames}")
print(f"Elapsed time (seconds): {elapsed_time:.2f}")
print(f"Average FPS: {total_frames / elapsed_time:2f}")

# nyc_night.mp4
# Total frames processed: 5245 
# Elapsed time (seconds): 134.92 
# Average FPS: 38.874242

# nyc_night.mp4, verbose=False
# Total frames processed: 5245
# Elapsed time (seconds): 113.05
# Average FPS: 46.394564

# bratislava_night.mp4
# Total frames processed: 16529
# Elapsed time (seconds): 828.75 
# Average FPS: 19.944441

# bratislava_night.mp4, verbose=False
# Total frames processed: 16529
# Elapsed time (seconds): 821.82
# Average FPS: 20.112688

# bratislava_night_640.mp4, verbose=False
# Total frames processed: 16529
# Elapsed time (seconds): 382.09 
# Average FPS: 43.258984

# test_video_android.mp4, verbose=False
# Total frames processed: 2961
# Elapsed time (seconds): 57.30
# Average FPS: 51.676606

# test_video_android.mp4, slicing
# Total frames processed: 2961
# Elapsed time (seconds): 1196.30
# Average FPS: 2.475141
