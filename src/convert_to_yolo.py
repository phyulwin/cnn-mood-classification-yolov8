#!/usr/bin/env python3

# defines YOLO train and test, and validation sets

import os, shutil
from PIL import Image

# adjust these paths to raw data
RAW_DIR   = "data/raw/archive/DATASET"
OUT_IMGS  = "data/processed/images"
OUT_LABEL = "data/processed/labels"
SPLITS    = {"train":"train", "val":"test"}  # use test as val

for split, raw_split in SPLITS.items():
    # create output folders
    src_root = os.path.join(RAW_DIR, raw_split)
    dst_img  = os.path.join(OUT_IMGS,  split)
    dst_lbl  = os.path.join(OUT_LABEL, split)
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    for class_id in map(str, range(1,8)):      # folders "1"…"7"
        src_cls = os.path.join(src_root, class_id)
        for fn in os.listdir(src_cls):
            if not fn.lower().endswith((".jpg",".png")): continue

            # copy image
            src_img = os.path.join(src_cls, fn)
            dst_img_f = os.path.join(dst_img, fn)
            shutil.copy(src_img, dst_img_f)

            # full‐frame bbox: x_center,y_center=0.5,0.5, width,height=1,1
            label_txt = f"{int(class_id)-1} 0.5 0.5 1.0 1.0\n"
            txt_name = os.path.splitext(fn)[0] + ".txt"
            with open(os.path.join(dst_lbl, txt_name), "w") as f:
                f.write(label_txt)

    print(f"Converted {split} → {dst_img} + {dst_lbl}")