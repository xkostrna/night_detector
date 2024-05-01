from pathlib import Path

labels_pth = Path('/home/xkostrna/night-detector/datasets/exdark/yolo/exdark640bg/train/labels')
for lbl in labels_pth.iterdir():
    if 'bg' in lbl.stem:
        lines = ["12 0.5 0.5 1.0 1.0"]
        lbl.open('w', encoding='utf-8').writelines(lines)
