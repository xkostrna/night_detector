from pathlib import Path

dataset_pth = Path("/home/xkostrna/night-detector/datasets/exdark/yolo/keys_v2")

img_sets = ["train", "test", "valid"]

for img_set in img_sets:
    for lbl_pth in (dataset_pth / img_set / 'labels').iterdir():
        lines = lbl_pth.open('r', encoding='utf-8').readlines()
        new_lines = [line.replace("0", "12", 1) for line in lines]
        lbl_pth.open('w', encoding='utf-8').writelines(new_lines)
