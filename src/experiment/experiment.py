from pathlib import Path
from shutil import copy
from typing import Union
from collections import defaultdict

from numpy import ndarray
from cv2 import imread, imwrite
import albumentations

from src.utils import dump_yolo_bboxes, get_yolo_labels


TRANSFORMATOR_1 = albumentations.Compose(
    transforms=[albumentations.RandomBrightnessContrast(brightness_limit=(0.1, 0.2),
                                                        contrast_limit=(0.1, 0.15),
                                                        brightness_by_max=True,
                                                        p=1)],
    bbox_params=albumentations.BboxParams(format='yolo',
                                          label_fields=['category_ids'])
)


def create_dataset_by_exdark(exdark_pth: Path,
                             dest_pth: Path,
                             class_list_pth: Path,
                             class_filter: Union[list[int], None] = None):
    """Creates dataset by using exdark imageclasslist.txt which serves for splitting images into train, test, valid.

    :param exdark_pth: path to exdark dataset
    :param dest_pth: path to destination for yolo dataset with train, test, valid folders prepared
    :param class_list_pth: path to imageclasslist.txt
    :param class_filter: optional parameter, can be used to filter out classes of the dataset
    """
    images = exdark_pth / 'images'
    labels = exdark_pth / 'labels'

    img_to_set = {1: dest_pth / 'train',
                  2: dest_pth / 'valid',
                  3: dest_pth / 'test'}

    # every array indexing in this loop is defined by exdark imageclasslist.txt
    lines = class_list_pth.open(mode='r', encoding='utf-8').readlines()
    for line in lines:
        parts = line.split(' ')
        img_pth = images / parts[0]
        if not img_pth.exists():
            # img_pth = img_pth.with_suffix('.jpg')
            continue

        label_pth = (labels / parts[0]).with_suffix('.txt')
        dest = img_to_set[int(parts[4])]

        if not class_filter:
            copy(img_pth, dest / 'images')
            copy(label_pth, dest / 'labels')
            continue

        class_ids, bboxes = filter_yolo_classes(label_pth, class_filter)
        if (class_ids, bboxes) != (-1, -1):
            copy(img_pth, dest / 'images')
            dump_yolo_bboxes(label_pth=dest / 'labels' / label_pth.name,
                             class_ids=class_ids,
                             bboxes=bboxes)


def filter_yolo_classes(label_pth: Path, class_filter: list[int]) -> tuple:
    """Class filtering from annotations file.

    Deletes lines starting with id not found in class_filter, if id is found then its swapped by position
    because yolo needs class ids to start from 0
    """
    class_ids, bboxes = [], []

    for line in label_pth.open(mode='r', encoding='utf-8').readlines():
        parts = line.strip().split(' ')
        class_id, *bbox = parts
        class_id = int(class_id)
        # with this condition we completely skips images where was previously some object
        if class_id not in class_filter:
            return -1, -1
        # if class_id in class_filter:
            # if we want whole new dataset we have to use index(class_id)
            # if we want to create dataset to augment other dataset then it must be exact class_id
        class_ids.append(class_id)
        bboxes.append(bbox)

    if len(class_ids) == 0 or len(bboxes) == 0:
        return -1, -1

    return class_ids, bboxes


def augment_image(image_pth: Path, label_pth: Path) -> (ndarray, list[int], list):
    """Use albumentations Transformator to augment image and return transformed image, class ids and bboxes."""
    global TRANSFORMATOR_1
    image = imread(str(image_pth))
    class_ids, bboxes = get_yolo_labels(label_pth=label_pth)
    transformed = TRANSFORMATOR_1(image=image, bboxes=bboxes, category_ids=class_ids)
    return transformed['image'], class_ids, transformed['bboxes']


def augment_yolo_dataset(dataset_path: Path) -> None:
    """Iterate through train set of dataset, augment every image and save with according labels."""
    images = dataset_path / 'train' / 'images'
    labels = dataset_path / 'train' / 'labels'

    for idx, (img_pth, label_pth) in enumerate(zip(images.iterdir(), labels.iterdir())):
        try:
            image, class_ids, bboxes = augment_image(img_pth, label_pth)
            new_image_pth = img_pth.with_name(f'{idx}.jpg')
            new_label_pth = label_pth.with_name(f'{idx}.txt')
            imwrite(filename=str(new_image_pth), img=image)
            dump_yolo_bboxes(new_label_pth, class_ids, bboxes)
        except ValueError as e:
            print(f'[AUGMENTATION ERROR] {e}')
            continue


def balance_dataset(old: Path, new: Path, instances_limit: int):
    """This function creates new dataset from old by decreasing all objects instances down to limit."""
    class_counter = defaultdict(int)
    for img_pth, label_pth in zip((old / 'images').iterdir(), (old / 'labels').iterdir()):
        class_ids, bboxes = get_yolo_labels(label_pth)
        new_ids, new_bboxes = [], []
        for class_id, bbox in zip(class_ids, bboxes):
            if class_counter[class_id] >= instances_limit:
                continue
            class_counter[class_id] += 1
            new_ids.append(class_id)
            new_bboxes.append(bbox)

        if len(new_ids) == 0:
            continue

        copy(img_pth, new / 'images')
        dump_yolo_bboxes(new / 'labels' / label_pth.name, new_ids, new_bboxes)
    print('Balancing finished')


def main():
    # Bicycle(1), Boat(2), Bottle(3), Bus(4), Car(5), Cat(6),
    # Chair(7), Cup(8), Dog(9), Motorbike(10), People(11), Table(12)
    # class_filter = [4]
    create_dataset_by_exdark(exdark_pth=Path('../../datasets/exdark/undivided/balanced640'),
                             dest_pth=Path('../../datasets/exdark/yolo/exdark640balanced'),
                             class_list_pth=Path('imageclasslist.txt'))
    # balance_dataset(old=Path("../../datasets/exdark/undivided/default640"),
    #                 new=Path("../../datasets/exdark/undivided/balanced640"),
    #                 instances_limit=706)

    # augment_image(Path('../../datasets/exdark-yolo/exdark-yolo-green/train/images/2015_00002.png'),
    #               Path('../../datasets/exdark-yolo/exdark-yolo-green/train/labels/2015_00002.txt'))
    # augment_yolo_dataset(Path('../../datasets/exdark/yolo/balanced'))

    # balance_dataset(old=Path('../../datasets/exdark-yolo'),
    #                 new=Path('../../datasets/exdark-yolo/exdark-yolo-balanced'),
    #                 instances_limit=703)


if __name__ == "__main__":
    main()
