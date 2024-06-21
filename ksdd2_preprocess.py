import os
import cv2
from glob import glob
import pandas as pd
import argparse
from tqdm import tqdm


def reshape_ksdd2(src_dir, dst_dir, RES=(224, 632)):
    # make dest directory
    splits = ['train', 'test']
    for split in splits:
        src_split_dir = os.path.join(src_dir, split)
        dst_split_dir = os.path.join(dst_dir, split)
        os.makedirs(dst_split_dir, exist_ok=True)
        all_imgs = os.listdir(src_split_dir)
        for img in tqdm(all_imgs, desc=f"Reshaping {split} images", unit="file", total=len(all_imgs)):
            img_path = os.path.join(src_split_dir, img)
            img_out_path = os.path.join(dst_split_dir, img)
            img = cv2.imread(img_path)
            img = cv2.resize(img, RES)
            cv2.imwrite(img_out_path, img)

def copy_files(src_dir, dst_dir):
    target_files = glob(os.path.join(src_dir, '*.pyb'))
    for file in tqdm(target_files, desc="Copying .pyb files", unit="file", total=len(target_files)):
        file_name = os.path.basename(file)
        dst_file = os.path.join(dst_dir, file_name)
        os.system(f'cp {file} {dst_file}')

def make_csv(dst_dir):
    splits = ['train', 'test']
    for split in splits:
        img_dir = os.path.join(dst_dir, split)
        all_imgs = os.listdir(img_dir)
        all_masks = [img for img in all_imgs if "GT" in img]
        imgs_dict = {"path": [], "label": []}
        for img in tqdm(all_masks, desc=f"Creating {split}.csv", unit="file", total=len(all_masks)):
            imgs_dict["path"].append(img.replace("_GT.png", ".png"))
            img_path = os.path.join(img_dir, img)
            loaded = cv2.imread(img_path)
            # if there is a 1, it is positive, else negative
            if max(loaded.flatten()) == 0:
                imgs_dict["label"].append("negative")
            else:
                imgs_dict["label"].append("positive")
        df = pd.DataFrame(imgs_dict)
        df.to_csv(os.path.join(dst_dir, f"{split}.csv"), index=False)

def main(args):
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    RES = (224,632) # w x h
    # make directory
    print(f"Copying files from {src_dir} to {dst_dir}")
    os.makedirs(dst_dir, exist_ok=True)
    # copy .pyb files
    copy_files(src_dir, dst_dir)
    # reshape images (needed for batching)
    reshape_ksdd2(src_dir, dst_dir, RES=RES)
    # make csv files
    make_csv(dst_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True, help="Path to the KSDD2 dataset root")
    parser.add_argument("--dst_dir", type=str, required=True, help="Path to the destination directory")
    args = parser.parse_args()
    main(args)