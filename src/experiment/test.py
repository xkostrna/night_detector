from pathlib import Path
from collections import defaultdict

from src.utils import get_yolo_labels

# EXDARK SPECIFIC !
CLASS_ID_TO_NAME = {0: "Bicycle",
                    1: "Boat",
                    2: "Bottle",
                    3: "Bus",
                    4: "Car",
                    5: "Cat",
                    6: "Chair",
                    7: "Cup",
                    8: "Dog",
                    9: "Motorbike",
                    10: "People",
                    11: "Table"}


def yolo_class_instance_check(pth: Path):
    """Check number of instances per class."""
    instance_counter = defaultdict(int)
    for label_pth in (pth / 'labels').iterdir():
        class_ids, _ = get_yolo_labels(label_pth)
        for class_id in class_ids:
            instance_counter[class_id] += 1
    return instance_counter


def yolo_class_images_check(pth: Path):
    """Check number of images per class"""
    images_counter = defaultdict(int)
    for label_pth in (pth / 'labels').iterdir():
        class_ids, _ = get_yolo_labels(label_pth)
        for class_id in set(class_ids):
            images_counter[class_id] += 1
    return images_counter


def is_ok(dataset_pth: Path):
    """Check if every image has label."""
    for img_set in [pth for pth in dataset_pth.iterdir() if pth.is_dir()]:
        for img, label in zip((img_set / 'images').iterdir(), (img_set / 'labels').iterdir()):
            if img.stem != label.stem:
                print(f"Img name {img.stem} doesn't equal label name {label.stem}")
            if len(label.open('r').readlines()) < 1:
                print(f"Found empty label file: {label}")


def main():
    # print(yolo_class_images_check(Path("../../datasets/exdark/undivided/default640")))
    # print(yolo_health_check(Path('../../datasets/exdark/undivided/balanced640')))
    # instances = {0: 1120, 4: 2927, 3: 706, 6: 2377, 10: 7460, 9: 1072, 8: 1017, 11: 1483, 2: 1593, 1: 1389, 5: 909, 7: 1656}
    # images = {0: 751, 4: 1312, 3: 570, 6: 1230, 10: 2658, 9: 587, 8: 856, 11: 995, 2: 706, 1: 694, 5: 763, 7: 841}
    # sorted_dict = dict(sorted(images.items(), key=lambda item: item[1]))
    #
    # sum_up = 0
    # for k, v in sorted_dict.items():
    #     print(CLASS_ID_TO_NAME[k], v)
    #     sum_up += v
    # print(sum_up)
    print(yolo_class_instance_check(Path("../../datasets/exdark/undivided/balanced640")))
    # print(yolo_class_instance_check(Path("../../datasets/keys/keys_v2_aug/test")))
    # print(yolo_class_instance_check(Path("../../datasets/keys/keys_v2_aug/valid")))

    # print(yolo_health_check(Path("../../datasets/exdark/undivided/default640")))
    # print(is_ok(Path('../../datasets/exdark/yolo/balanced')))
    # a = yolo_health_check(Path('../../datasets/exdark/yolo/exdark-enlighten-640/train'))
    # b = yolo_health_check(Path('../../datasets/exdark/yolo/exdark-enlighten-640-no-people/train'))
    # print(f"Train set instances of ExDark: {sorted(a.items())}")
    # print(f"Train set instances of ExDark without people: {sorted(b.items())}")
    #
    # c = {}
    # for key, val in a.items():
    #     if key in b:
    #         c[key] = val + b[key]
    #     else:
    #         c[key] = val
    #
    # print(f"Train set instances of ExDark after adding 1000 images without people: {sorted(c.items())}")
    #
    # is_ok(Path('../../datasets/exdark/yolo/exdark640-augm'))


if __name__ == "__main__":
    main()
