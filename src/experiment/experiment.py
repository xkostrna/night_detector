from pathlib import Path
from shutil import copy
from typing import Union

from cv2 import imread, imwrite
import albumentations

from visualize import visualize_yolo_bboxes
from src.utils import dump_yolo_bboxes


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

    lines = label_pth.open(mode='r', encoding='utf-8').readlines()
    for line in lines:
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


def augment_dataset(dataset_path: Path, category_id_to_name: dict[int, str]):
    target_path = Path("F:/School/Ing/DIPLOMA/night_detector/datasets/exdark-yolo/exdark-yolo-green/augment-test")
    target_imgs = target_path / 'images'
    target_labels = target_path / 'labels'

    for image_set in dataset_path.iterdir():
        if not image_set.is_dir():
            continue

        images_pth = image_set / 'images'
        labels_pth = image_set / 'labels'

        transform = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
        ],
            bbox_params=albumentations.BboxParams(format='yolo', label_fields=['category_ids']),
        )

        for i, (img, label) in enumerate(zip(images_pth.iterdir(), labels_pth.iterdir())):
            lines = label.open(mode='r', encoding='utf-8').readlines()
            category_ids = []
            bboxes = []
            image = imread(str(img))
            for line in lines:
                parts = line.strip().split(' ')
                category_ids.append(int(parts[0]))
                bboxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

            transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
            print(fr'{target_imgs}\{i}.jpg')
            imwrite(filename=str(fr'{target_imgs}\{i}.jpg'), img=transformed['image'])
            target_label = target_labels / f'{i}.txt'
            # with target_label.open(mode='w', encoding='utf-8') as target:
            dump_yolo_bboxes(target_label, category_ids, transformed['bboxes'])
            for category_id, bbox in zip(category_ids, transformed['bboxes']):
                # target.write(cat)
                print(str(bbox))
            # target_label.open(mode='w', encoding='utf-8').writelines(bboxes)
            print(i, img, label)
            exit(-1)


def main():
    class_filter = [2, 4, 6, 7]
    create_dataset_by_exdark(exdark_pth=Path('../../datasets/exdark-yolo'),
                             dest_pth=Path('../../datasets/exdark-yolo/exdark-yolo-green'),
                             class_list_pth=Path('imageclasslist.txt'),
                             class_filter=class_filter)


if __name__ == "__main__":
    main()
