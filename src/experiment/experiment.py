from pathlib import Path
from shutil import copy
from typing import Union

import numpy
from cv2 import imread, imwrite
import albumentations

from visualize import visualize_yolo_bboxes
from src.utils import dump_yolo_bboxes, get_yolo_labels

TRANSFORMATOR_1 = albumentations.Compose(transforms=[albumentations.HorizontalFlip(p=1),
                                                     albumentations.ShiftScaleRotate(p=1),
                                                     albumentations.RGBShift(r_shift_limit=30,
                                                                             g_shift_limit=30,
                                                                             b_shift_limit=30, p=1)],
                                         bbox_params=albumentations.BboxParams(format='yolo',
                                                                               label_fields=['category_ids']))

TRANSFORMATOR_2 = albumentations.Compose(transforms=[albumentations.VerticalFlip(p=1),
                                                     albumentations.ShiftScaleRotate(p=1)],
                                         bbox_params=albumentations.BboxParams(format='yolo',
                                                                               label_fields=['category_ids']))


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
    lines = class_list_pth.open(mode='r', encoding='utf-8').readlines()
    images = exdark_pth / 'images'
    labels = exdark_pth / 'labels'

    img_to_set = {1: dest_pth / 'train',
                  2: dest_pth / 'valid',
                  3: dest_pth / 'test'}

    # every array indexing in this loop is defined by exdark imageclasslist.txt
    for line in lines:
        parts = line.split(' ')
        img_pth = images / parts[0]
        label_pth = (labels / parts[0]).with_suffix('.txt')

        can_be_copied = filter_yolo_classes(label_pth, class_filter)
        if not can_be_copied:
            continue

        dest = img_to_set[int(parts[4])]
        copy(img_pth, dest / 'images')
        copy(label_pth, dest / 'labels')


def filter_yolo_classes(label_pth: Path, class_filter: Union[list[int], None] = None) -> bool:
    """Class filtering from annotations file.

    Deletes lines starting with id not found in class_filter, if id is found then its swapped by position
    because yolo needs class ids to start from 0
    """
    if not class_filter:
        return True

    class_ids, bboxes = [], []

    for line in label_pth.open(mode='r', encoding='utf-8').readlines():
        parts = line.strip().split(' ')
        class_id, *bbox = parts
        class_id = int(class_id)
        if class_id in class_filter:
            class_ids.append(class_filter.index(class_id))
            bboxes.append(bbox)

    if len(class_ids) == 0 or len(bboxes) == 0:
        return False

    dump_yolo_bboxes(label_pth=label_pth, class_ids=class_ids, bboxes=bboxes)
    return True


def augment_image(image_pth: Path, label_pth: Path) -> (numpy.ndarray, list[int], list):
    """Use albumentations Transformator to augment image and return transformed image, class ids and bboxes."""
    global TRANSFORMATOR_1, TRANSFORMATOR_2
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


def main():
    # class_filter = [4]
    # create_dataset_by_exdark(exdark_pth=Path('../../datasets/exdark-yolo'),
    #                          dest_pth=Path('../../datasets/exdark-yolo/exdark-yolo-caronly'),
    #                          class_list_pth=Path('imageclasslist.txt'),
    #                          class_filter=class_filter)

    # augment_image(Path('../../datasets/exdark-yolo/exdark-yolo-green/train/images/2015_00002.png'),
    #               Path('../../datasets/exdark-yolo/exdark-yolo-green/train/labels/2015_00002.txt'))
    augment_yolo_dataset(Path('../../datasets/exdark-yolo/exdark-yolo-caronly'))


if __name__ == "__main__":
    main()
