from pathlib import Path

import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/exdark640key/train-16b-32e/weights/best.pt")
# BGR format cause cv2
BOX_COLOR_BLUE = (255, 21, 25)
BOX_COLOR_RED = (54, 54, 255)
TEXT_COLOR = (255, 255, 255)


def get_yolo_labels(label_pth: Path) -> (list[int], list[list[float]]):
    class_ids, bboxes = [], []
    for line in label_pth.open(mode='r', encoding='utf-8').readlines():
        parts = line.strip().split(' ')
        class_id, *bbox = parts
        class_ids.append(int(class_id))
        bboxes.append([float(num) for num in bbox])
    return class_ids, bboxes


def yolo2pixels(img_height, img_width, bbox: list[float]) -> tuple[int, int, int, int]:
    """Convert normalized YOLO coordinates to pixel coordinates"""
    x_center, y_center, w, h = bbox
    x_min = int((x_center - w / 2) * img_width)
    y_min = int((y_center - h / 2) * img_height)
    x_max = int((x_center + w / 2) * img_width)
    y_max = int((y_center + h / 2) * img_height)
    return x_min, y_min, x_max, y_max


def visualize_yolo_bboxes(img,
                          bboxes: list[list[float]],
                          class_ids: list[int],
                          class_id_to_name: dict[int, str],
                          prob: list[float],
                          box_color: tuple[int, int, int]):
    height, width, _ = img.shape

    for bbox, category_id in zip(bboxes, class_ids):
        class_name = class_id_to_name[category_id]
        x_min, y_min, x_max, y_max = yolo2pixels(height, width, bbox)

        # Draw bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=box_color, thickness=2)
        class_name = f"{class_name} {prob[0]:.2f}" if prob else class_name

        # Draw class name
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), box_color, -1)
        cv2.putText(img,
                    text=class_name,
                    org=(x_min, y_min - int(0.3 * text_height)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75,
                    color=TEXT_COLOR,
                    lineType=cv2.LINE_AA)
    return img


def visualize_yolo_bboxes_ai(img,
                             bboxes: list[list[float]],
                             class_ids: list[int],
                             class_id_to_name: dict[int, str],
                             probs: list[float],
                             box_color: tuple[int, int, int]):
    height, width, _ = img.shape

    for bbox, category_id, prob in zip(bboxes, class_ids, probs):
        class_name = class_id_to_name[category_id]
        x_min, y_min, x_max, y_max = yolo2pixels(height, width, bbox)

        # Draw bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=box_color, thickness=2)
        class_name = f"{class_name} {prob:.2f}" if prob else class_name

        # Draw class name
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)
        cv2.rectangle(img, (x_min, y_max), (x_min + text_width, y_max - int(1.3 * text_height)), box_color, -1)
        cv2.putText(img,
                    text=class_name,
                    org=(x_min, y_max - int(0.3 * text_height)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75,
                    color=TEXT_COLOR,
                    lineType=cv2.LINE_AA)
    return img


def main():
    class_names = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike', 'People',
                   'Table', 'Key']
    class_ids_to_name = dict(zip([num for num in range(0, 13)], class_names))
    dataset_pth = Path("datasets/exdark/yolo/exdark640testonly")

    # i = 0
    for i, img_pth in enumerate((dataset_pth / "images").iterdir()):
        label_pth = dataset_pth / "labels" / f"{img_pth.stem}.txt"
        class_ids, bboxes = get_yolo_labels(label_pth)
        img = cv2.imread(str(img_pth))
        img = visualize_yolo_bboxes(img, bboxes, class_ids, class_ids_to_name, [], BOX_COLOR_BLUE)

        results = model.predict(img_pth, save=False, conf=0.5)
        bboxes = results[0].boxes.xywhn.tolist()
        probs = results[0].boxes.conf.tolist()
        class_ids = [int(_id) for _id in results[0].boxes.cls.tolist()]
        img = visualize_yolo_bboxes_ai(img, bboxes, class_ids, class_ids_to_name, probs, BOX_COLOR_RED)

        cv2.imwrite(f'runs/predict/multiple/result{i}.jpg', img)
        # i += 1
        # if i == 50:
        #     break


if __name__ == '__main__':
    main()
