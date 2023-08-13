from pathlib import Path
from typing import Union

from pybboxes import convert_bbox
import cv2

YOLO_SIZE = 416
BBOX_SIZE = 4


def parse_bbgt_line(line: str) -> tuple[str, int, int, int, int]:
    """Parse raw line from bbgt label file.

    more details: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Groundtruth
    l - pixel number from left of image
    t - pixel number from top of image
    w - width of bounding box
    h - height of bounding box
    """
    class_name, left, top, width, height = line.split(' ')[:5]  # only first 5 items are important
    left, top, width, height = int(left), int(top), int(width), int(height)
    return class_name, left, top, width, height


def rescale_bbox(bbox: tuple, vsf: float, hsf: float) -> Union[tuple, None]:
    """Rescales bbox using vsf and hsf and return's new bbox.

    @param bbox bounding box, usually has 4 numbers inside
    @param vsf vertical scale factor
    @param hsf horizontal scale factor
    """
    if len(bbox) != BBOX_SIZE:
        return None
    left, top, orig_width, orig_height = bbox
    x_min = left * hsf
    y_min = top * vsf
    width = orig_width * hsf
    height = orig_height * vsf
    return x_min, y_min, width, height


def bbgt2yolo_format(image_pth: Path, label_pth: Path, img_classes: dict[str:int], yolo_pth: Path) -> None:
    """Transform bbgt label format to yolo format."""
    img = cv2.imread(str(image_pth))
    vertical_scale_factor = YOLO_SIZE / img.shape[0]  # height
    horizontal_scale_factor = YOLO_SIZE / img.shape[1]  # width
    resized_img = cv2.resize(img, (YOLO_SIZE, YOLO_SIZE))

    class_ids, bboxes = [], []
    lines = label_pth.open(mode='r+', encoding='utf-8').readlines()[1:]  # ignore comment in the file on the first line
    for line in lines:
        class_name, *bbox = parse_bbgt_line(line)
        class_ids.append(img_classes[class_name])
        bbox = rescale_bbox(bbox=bbox, vsf=vertical_scale_factor, hsf=horizontal_scale_factor)
        try:
            bboxes.append(convert_bbox(bbox=bbox, from_type="coco", to_type="yolo", image_size=(YOLO_SIZE, YOLO_SIZE)))
        except (ValueError, TypeError) as e:
            print(f"[CONVERT_ERROR] error: {e}, image_pth: {image_pth}, bbox: {bbox}")
            continue

    cv2.imwrite(filename=str(yolo_pth / 'images' / f"{image_pth.name}"), img=resized_img)

    label_pth_parts = label_pth.name.split('.')
    dump_yolo_bboxes(label_pth=yolo_pth / 'labels' / f"{label_pth_parts[0]}.{label_pth_parts[2]}",
                     class_ids=class_ids,
                     bboxes=bboxes)


def exdark2yolo(exdark_pth: Path, yolo_pth: Path) -> None:
    """Transform bbgt labels format to yolo format."""
    img_classes: dict[str:int] = {}
    images_pth = exdark_pth / 'images'
    labels_pth = exdark_pth / 'labels'

    for class_id, class_dir in enumerate(images_pth.iterdir()):
        img_classes[class_dir.name] = class_id

    for dir_id, (image_dir, label_dir) in enumerate(zip(images_pth.iterdir(), labels_pth.iterdir())):
        for img_id, (image_pth, label_pth) in enumerate(zip(image_dir.iterdir(), label_dir.iterdir())):
            bbgt2yolo_format(image_pth, label_pth, img_classes, yolo_pth)


def dump_yolo_bboxes(label_pth: Path, class_ids: list[int], bboxes: list[tuple]) -> None:
    """Dumps class ids with according bboxes into file using yolo format."""
    yolo_lines = [f"{class_id} {' '.join(str(num) for num in bbox)}\n" for class_id, bbox in zip(class_ids, bboxes)]
    label_pth.open(mode='w', encoding='utf-8').writelines(yolo_lines)


if __name__ == "__main__":
    exdark2yolo(exdark_pth=Path("../datasets/exdark"),
                yolo_pth=Path("../datasets/exdark-yolo"))
