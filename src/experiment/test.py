from pathlib import Path
from collections import defaultdict

from src.utils import get_yolo_labels


def yolo_health_check(pth: Path):
    """Check number of instances per class."""
    class_counter = defaultdict(int)
    for label_pth in (pth / 'labels').iterdir():
        class_ids, _ = get_yolo_labels(label_pth)
        for class_id in class_ids:
            class_counter[class_id] += 1
    print(sorted(class_counter.items()))


def is_ok(dataset_pth: Path):
    """Check if every image has label."""
    for img_set in [pth for pth in dataset_pth.iterdir() if pth.is_dir()]:
        for img, label in zip((img_set / 'images').iterdir(), (img_set / 'labels').iterdir()):
            if img.stem != label.stem:
                print(f"Img name {img.stem} doesn't equal label name {label.stem}")


def main():
    # yolo_health_check(Path('../../datasets/exdark/undivided/default'))
    is_ok(Path('../../datasets/exdark/yolo/exdark640'))


if __name__ == "__main__":
    main()
