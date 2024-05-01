from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8x-world.pt")
    model.set_classes(['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'])

    metrics_val = model.val(data="/home/xkostrna/night-detector/datasets/exdark/yolo/exdark640enlighten/data.yaml", split="val")
    metrics_test = model.val(data="/home/xkostrna/night-detector/datasets/exdark/yolo/exdark640enlighten/data.yaml", split="test")
    print(f"val mAP50-95: {metrics_val.results_dict['metrics/mAP50-95(B)']}")
    print(f"test mAP50-95: {metrics_test.results_dict['metrics/mAP50-95(B)']}")
