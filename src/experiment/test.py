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


def main():
    yolo_health_check(Path('../../datasets/exdark/undivided/default'))


if __name__ == "__main__":
    main()
