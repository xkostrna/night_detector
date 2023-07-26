from pathlib import Path
from pybboxes import convert_bbox
import cv2

YOLO_WIDTH = 640
YOLO_HEIGHT = 640


def parse_bbgt_line(line: str) -> tuple[str, int, int, int, int]:
    """Parse raw line from bbgt label file.

    l - pixel number from left of image
    t - pixel number from top of image
    w - width of bounding box
    h - height of bounding box
    """
    class_name, left, top, width, height = line.split(' ')[:5]
    left, top, width, height = int(left), int(top), int(width), int(height)
    return class_name, left, top, width, height


def bbgt2yolo_format(image_pth: Path,
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

    with label_pth.open(mode='r+', encoding='utf-8') as label_file:
        lines = label_file.readlines()[1:]  # without comment on first line

    new_labels = []
    for line in lines:
        class_name, left, top, width, height = parse_bbgt_line(line)  # only first 5 items are important
        class_id = image_classes[class_name]

        x_min = int(left * horizontal_scale_factor)
        y_min = int(top * vertical_scale_factor)
        width = width * horizontal_scale_factor
        height = height * vertical_scale_factor

        bbox = (x_min, y_min, width, height)
        try:
            center_x, center_y, width, height = convert_bbox(bbox=bbox,
                                                             from_type="coco",
                                                             to_type="yolo",
                                                             image_size=(YOLO_WIDTH, YOLO_HEIGHT))
            new_labels.append(f"{class_id} {center_x} {center_y} {width} {height}\n")
        except ValueError as e:
            print(e)
            print(image_pth)
            print(bbox)
            continue

    cv2.imwrite(filename=str(yolo_pth / 'images' / f"{dir_id}_{img_id}.jpg"), img=resized_img)
    dest_pth = yolo_pth / 'labels' / f"{dir_id}_{img_id}.txt"
    dest_pth.open(mode='w', encoding='utf-8').writelines(new_labels)


def bbgt2yolo(bbgt_pth: Path, yolo_pth: Path) -> None:
    """Transform bbgt labels format to yolo format.

    usage: bbgt2yolo(Path("datasets/exdark"), Path("datasets/exdark-yolo"))
    """
    image_classes: dict[int: str] = {}
    image_class_id = 0

    images_pth = bbgt_pth / 'images'
    labels_pth = bbgt_pth / 'labels'

    for image_dir in images_pth.iterdir():
        image_classes[image_dir.name] = image_class_id
        image_class_id += 1

    for dir_id, (image_dir, label_dir) in enumerate(zip(images_pth.iterdir(), labels_pth.iterdir())):
        for img_id, (image_pth, label_pth) in enumerate(zip(image_dir.iterdir(), label_dir.iterdir())):
            bbgt2yolo_format(image_pth, label_pth, image_classes, yolo_pth, dir_id, img_id)
