from pathlib import Path
from enum import Enum
from pybboxes import convert_bbox
import cv2

YOLO_WIDTH = 640
YOLO_HEIGHT = 640


def bbgt2yolo_label(image_pth: Path,
                    label_pth: Path,
                    image_classes: dict[int:str],
                    yolo_pth: Path,
                    dir_id: int,
                    img_id: int) -> None:
    """Parse bbgt label format to yolo format."""
    img = cv2.imread(str(image_pth))

    vertical_scale_factor = YOLO_HEIGHT / img.shape[0]  # height
    horizontal_scale_factor = YOLO_WIDTH / img.shape[1]  # width

    resized_img = cv2.resize(img, (YOLO_WIDTH, YOLO_HEIGHT))

    new_labels = []
    flag = 0
    with label_pth.open(mode='r+', encoding='utf-8') as label_file:
        lines = label_file.readlines()[1:]  # without comment on first line
        for line in lines:
            class_name, x_min, y_min, x_max, y_max = line.split(' ')[:5]  # only first 5 items are important
            class_id = image_classes[class_name]

            x_min = int(int(x_min) * horizontal_scale_factor)
            y_min = int(int(y_min) * vertical_scale_factor)
            x_max = int(int(x_max) * horizontal_scale_factor)
            y_max = int(int(y_max) * vertical_scale_factor)

            bbox = (x_min, y_min, x_max, y_max)
            try:
                center_x, center_y, width, height = convert_bbox(bbox=bbox,
                                                                 from_type="voc",
                                                                 to_type="yolo",
                                                                 image_size=(YOLO_WIDTH, YOLO_HEIGHT))
                new_labels.append(f"{class_id} {center_x} {center_y} {width} {height}\n")
                cv2.imwrite(filename=str(yolo_pth / 'images' / f"{dir_id}_{img_id}.jpg"),
                            img=resized_img)
            except ValueError:
                print(bbox)
                flag = 1

    if flag == 0:
        with (yolo_pth / 'labels' / f"{dir_id}_{img_id}.txt").open(mode='w', encoding='utf-8') as res:
            res.writelines(new_labels)


def bbgt2yolo(bbgt_pth: Path, yolo_pth: Path) -> None:
    """Transform bbgt labels format to yolo format."""

    image_classes: dict[int: str] = {}
    image_class_id = 0

    images_pth = bbgt_pth / 'images'
    labels_pth = bbgt_pth / 'labels'

    for image_dir in images_pth.iterdir():
        image_classes[image_dir.name] = image_class_id
        image_class_id += 1

    for dir_id, (image_dir, label_dir) in enumerate(zip(images_pth.iterdir(), labels_pth.iterdir())):
        for img_id, (image_pth, label_pth) in enumerate(zip(image_dir.iterdir(), label_dir.iterdir())):
            bbgt2yolo_label(image_pth,
                            label_pth,
                            image_classes,
                            yolo_pth,
                            dir_id,
                            img_id)
        break
