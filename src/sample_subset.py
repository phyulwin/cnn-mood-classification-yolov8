#!/usr/bin/env python3

# create subset of 6000 images and train on them

import os
import random
import shutil

def sample_flat(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, n=6000, seed=1234):
    random.seed(seed)
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    # list all image files
    all_imgs = [f for f in os.listdir(src_img_dir)
                if f.lower().endswith(('.jpg', '.png'))]
    if n > len(all_imgs):
        raise ValueError(f"Requested {n}, but only {len(all_imgs)} images available.")

    sampled = random.sample(all_imgs, n)
    for fn in sampled:
        # copy image
        shutil.copy(os.path.join(src_img_dir, fn),
                    os.path.join(dst_img_dir, fn))
        # copy corresponding label (.txt)
        lbl_name = os.path.splitext(fn)[0] + '.txt'
        shutil.copy(os.path.join(src_lbl_dir, lbl_name),
                    os.path.join(dst_lbl_dir, lbl_name))

if __name__ == "__main__":
    sample_flat(
        src_img_dir="data/processed/images/train",
        src_lbl_dir="data/processed/labels/train",
        dst_img_dir="data/processed/images/train_small",
        dst_lbl_dir="data/processed/labels/train_small",
        n=6000
    )
    print("Sampled 6000 images to 'train_small'.")