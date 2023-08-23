from pathlib import Path
from collections import defaultdict

from src.utils import get_yolo_labels


def health_check(pth: Path):
    class_counter = defaultdict(int)
    for label_pth in (pth / 'labels').iterdir():
        class_ids, _ = get_yolo_labels(label_pth)
        for class_id in class_ids:
            class_counter[class_id] += 1
    print(class_counter)


def main():
    health_check(Path('../../datasets/exdark-yolo/exdark-yolo-balanced'))


if __name__ == "__main__":
    main()
