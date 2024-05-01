from pathlib import Path
import csv
from PIL import Image


def rename_files(pth: Path):
    # pth = Path("../datasets/exdark/undivided/enlighten-extracted-no-people")
    for i, (img, lbl) in enumerate(zip((pth / 'images').iterdir(), (pth / 'labels').iterdir())):
        # img.rename(img.with_stem(f"{img.stem}_ENGAN"))
        # lbl.rename(lbl.with_stem(f"{lbl.stem}_ENGAN"))
        img.rename(img.with_stem(f"bg{i}"))
        lbl.rename(lbl.with_stem(f"bg{i}"))


def rescale_images(imgs_path: Path):
    new_size = (640, 640)
    for img_pth in imgs_path.iterdir():
        img = Image.open(img_pth)
        resized_image = img.resize(new_size)
        resized_image.save(img_pth)


def find_best_map():
    detect_pth = Path("../runs/detect")
    runs_dirs = [x for x in detect_pth.iterdir() if x.is_dir()]
    csv.register_dialect('MyDialect', quotechar='"', skipinitialspace=True, quoting=csv.QUOTE_NONE, lineterminator='\n',
                         strict=True)
    for runs_dir in runs_dirs:
        best_mAP = 0.0
        pth = Path("")
        for run in runs_dir.iterdir():
            results = (run / 'results.csv')
            if results.exists():
                last = results.open('r', encoding='utf-8').readlines()[-1]
                mAP = float([num.strip() for num in last.split(',')][7])
                if mAP > best_mAP:
                    best_mAP = mAP
                    pth = run
        print(f"{pth} mAP50-95 = {best_mAP}")


def create_empty_labels():
    pth = Path(r"F:\School\Ing\DIPLOMA\night_detector\datasets\backgrounds\bg640")
    for img in (pth / "images").iterdir():
        lbl_pth = img.parent.parent / "labels" / f"{img.stem}.txt"
        lbl_pth.touch()


def main():
    rescale_images(Path("../datasets/keys/final"))
    # rescale_images(Path("../datasets/keys/LAKTIC_KLUCE"))
    # rescale_images(Path("../datasets/keys/TOMAS_KLUCE"))
    # rename_files(Path("../datasets/backgrounds/bg640"))


if __name__ == "__main__":
    main()
