from pathlib import Path
from shutil import copy
from typing import Union

from cv2 import imread
import albumentations

from visualize import visualize_yolo_bboxes

IMAGE_CLASSES_PTH = Path("F:/School/Ing/DIPLOMA/night_detector/src/eda/imageclasslist.txt")
EXDARK_IMAGES = Path("F:/School/Ing/DIPLOMA/night_detector/datasets/exdark-yolo/images")
EXDARK_LABELS = Path("F:/School/Ing/DIPLOMA/night_detector/datasets/exdark-yolo/labels")
DEST = Path("F:/School/Ing/DIPLOMA/night_detector/datasets/exdark-yolo/exdark-yolo-green")

DEST_DICT: dict[int, Path] = {1: DEST / "train",
                              2: DEST / "valid",
                              3: DEST / "test"}


def create_dataset(class_filter: Union[dict[int, int], None] = None):
    data = IMAGE_CLASSES_PTH.open(mode='r', encoding='utf-8').readlines()

    for txt in data:
        parts = txt.split(' ')
        img_pth = EXDARK_IMAGES / parts[0]
        label_pth = (EXDARK_LABELS / parts[0]).with_suffix('.txt')

        can_be_copied = filter_labels(label_pth, class_filter)

        if not can_be_copied:
            continue

        dest = DEST_DICT[int(parts[4])]
        copy(img_pth, dest / 'images')
        copy(label_pth, dest / 'labels')


def filter_labels(label_pth: Path, class_filter: Union[dict[int, int], None] = None) -> bool:
    if not class_filter:
        return True

    original_lines = label_pth.open(mode='r', encoding='utf-8').readlines()
    new_lines = []

    for line in original_lines:
        parts = line.split(' ')
        class_id, left, top, width, height = parts
        class_id = int(class_id)
        if class_id in class_filter:
            new_lines.append(f"{class_filter[class_id]} {left} {top} {width} {height}")

    if len(new_lines) == 0:
        return False

    label_pth.open(mode='w', encoding='utf-8').writelines(new_lines)
    return True


def augment_dataset(dataset_path: Path, category_id_to_name: dict[int, str]):
    for image_set in dataset_path.iterdir():
        if not image_set.is_dir():
            continue

        images_pth = image_set / 'images'
        labels_pth = image_set / 'labels'

        transform = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.RandomBrightnessContrast(p=0.3),
            albumentations.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
        ],
            bbox_params=albumentations.BboxParams(format='yolo', label_fields=['category_ids']),
        )

        for img, label in zip(images_pth.iterdir(), labels_pth.iterdir()):
            lines = label.open(mode='r', encoding='utf-8').readlines()
            category_ids = []
            bboxes = []
            image = imread(str(img))
            for line in lines:
                parts = line.strip().split(' ')
                category_ids.append(int(parts[0]))
                bboxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

            transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
            visualize_yolo_bboxes(transformed['image'],
                                  transformed['bboxes'],
                                  transformed['category_ids'],
                                  category_id_to_name)
            exit(-1)


def main():
    # class_filter = {2: 0,  # Bottle
    #                 4: 1,  # Car
    #                 6: 2,  # Chair
    #                 7: 3}  # Cup
    # create_dataset(class_filter)
    category_id_to_name = {0: 'Bottle', 1: 'Car', 2: 'Chair', 3: 'Cup'}
    dataset_path = Path("F:/School/Ing/DIPLOMA/night_detector/datasets/exdark-yolo/exdark-yolo-green")
    augment_dataset(dataset_path, category_id_to_name)


if __name__ == "__main__":
    main()
