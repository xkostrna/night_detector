import csv
import sys
from pathlib import Path

runs_pth = Path(str(sys.argv[1]))

model_dirs = [pth for pth in runs_pth.iterdir() if pth.is_dir()]

csv.register_dialect('MyDialect', quotechar='"', skipinitialspace=True, quoting=csv.QUOTE_NONE, lineterminator='\n', strict=True)

for model_dir in model_dirs:
    best_mAP = 0.0
    best_pth = Path('')
    for train in model_dir.iterdir():
        if train.is_dir():
            last = (train / 'results.csv').open('r', encoding='utf-8').readlines()[-1]
            mAP = float([num.strip() for num in last.split(',')][7])
            if mAP > best_mAP:
                best_mAP = mAP
                best_pth = train
    print(f"{best_pth} mAP50-95 = {best_mAP}")
