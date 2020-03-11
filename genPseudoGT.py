import numpy as np
import pandas
import os
import cv2

import torch
from torch.utils.data import Dataset, Dataloader

import image_proc

from matplotlib import pyplot as plt

class ICDAR2015(Dataset):
    def __init__(self, img_dir, gt_dir, color_flag=1):
        super(ICDAR2015).__init__()

        # paths
        self.img_dir = img_dir
        self.gt_dir  = gt_dir

        # flags
        self.color_flag = color_flag

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx += 1
        img_name = f"img_{idx}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path, self.color_flag)

        gt_name = f"gt_img_{idx}.txt"
        gt_path = os.path.join(self.gt_dir, gt_name)

        headers = ['x1','y1','x2','y2','x3','y3','x4','y4','transcription']
        gt_df  = pandas.read_csv(gt_path, names=headers)
        wordBBs = gt_df['x1','y1','x2','y2','x3','y3','x4','y4'].to_numpy().reshape(-1,4,2)
        texts = gt_df['transcription']

        return img, wordBBs, texts



class PseudoGTDataset(self):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if changing starting index
        idx += self.begin

        f = self.f

        # get image
        # for each wordBB, crop to c_i
        # for each c_i, perform inference
        # return inferred to a blank gt template
        # return img, gt

if __name__ == '__main__':
    base_dir = "C:/Users/Aerjay/Downloads/datasets/icdar-2015/"
    img_dir = os.path.join(base_dir, "4.1 Localization/ch4_training_images")
    img_dir = os.path.join(base_dir, "4.1 Localization/ch4_training_localization_transcription_gt")
    dataset = ICDAR2015(img_dir, gt_dir)

    img, wordBBs, texts = dataset[0]
    print(f"img.shape = {img.shape}")

    plt.figure()
    plt.imshow(img)
    plt.show()