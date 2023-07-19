from pathlib import Path
from enum import Enum
import cv2

YOLO_WIDTH = 640
YOLO_HEIGHT = 640


def bbgt2yolo_label(image_pth: Path, label_pth: Path, image_classes: dict[int:str]) -> str:
    img = cv2.imread(str(image_pth))
    resized_img = cv2.resize(img, (YOLO_WIDTH, YOLO_HEIGHT))
    cv2.imwrite(str(image_pth.with_name('test.jpg')), resized_img)
    return ''

    # with label_pth.open(mode='r+', encoding='utf-8') as label_file:
    #     lines = label_file.readlines()[1:]  # without comment on first line
    #     for line in lines:
    #         class_name, x_min, y_min, x_max, y_max = line.split(' ')[:5]  # only first 5 items are important
    #         # center_x = (x_min + x_max) / (2 * image_width)
    #         # center_y = (y_min + y_max) / (2 * image_height)
    #         # width = (x_max - x_min) / image_width
    #         # height = (y_max - y_min) / image_height
    # pass


def bbgt2yolo(bbgt_pth: Path, yolo_pth: Path):
    """Transform bbgt labels format to yolo format."""

    image_classes: dict[int: str] = {}
    image_class_id = 0

    images_pth = bbgt_pth / 'images'
    labels_pth = bbgt_pth / 'labels'

    for image_dir in images_pth.iterdir():
        image_classes[image_class_id] = image_dir.name.lower()
        image_class_id += 1

    for image_dir, label_dir in zip(images_pth.iterdir(), labels_pth.iterdir()):
        for image_pth, label_pth in zip(image_dir.iterdir(), label_dir.iterdir()):
            bbgt2yolo_label(image_pth, label_pth, image_classes)
            break
        break

    #
    #
    # elif fpath.name == 'labels':
    #     for labels_dir in fpath.iterdir():
    #         for label in labels_dir.iterdir():
    #             bbgt2yolo_label(label, image_classes)
    #             break


def resize_images(images_path: Path):
    for fpath in images_path.iterdir():
        pass
