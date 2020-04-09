import numpy as np
import pandas
import os
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from craft.craft import CRAFT

import image_proc

from matplotlib import pyplot as plt

class ICDAR2015Dataset(Dataset):
    def __init__(self, img_dir, gt_dir, color_flag=1):
        super(ICDAR2015Dataset).__init__()

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
        wordBBs = gt_df[['x1','y1','x2','y2','x3','y3','x4','y4']].to_numpy().reshape(-1,4,2)
        texts = gt_df['transcription']

        return img, wordBBs, texts



class PseudoGTDataset(ICDAR2015Dataset):
    def __init__(self, img_dir, gt_dir, weights_path=None, color_flag=1,
                 character_map=True, affinity_map=False, word_map=True,
                 direction_map=True):
        # super(ICDAR2015Dataset).__init__()
        self.raw_dataset = ICDAR2015Dataset(img_dir, gt_dir, color_flag=color_flag)

        self.num_class = character_map + affinity_map + word_map + 2*direction_map
        model = CRAFT(pretrained=False, num_class=self.num_class).cuda()
        if weights_path:
            model.load_state_dict(torch.load(weights_path))
            model.eval()
        self.model = model


    def __getitem__(self, idx):
        img, wordBBs, texts = self.raw_dataset[idx]

        # get image
        # for each wordBB, crop to c_i
        # for each c_i, perform inference
        # return inferred to a blank gt template
        # return img, gt

        model = self.model

        C,H,W = img.shape
        gt = np.zeros((H // 2, W // 2, self.num_class), dtype="float32")
        for wordBB in wordBBs:
            y_min, x_min = np.min(wordBB, axis=0)
            y_max, x_max = np.max(wordBB, axis=0)
            wordBB_img = img[:,y_min:y_max, x_min:x_max]

            # wordBB_gt,_ = model(wordBB_img)

            # y_min, x_min = int(y_min/2.), int(x_min/2.)
            # y_max = y_min + wordBB_gt.shape[0]
            # x_max = x_min + wordBB_gt.shape[1]
            # gt[y_min:y_max,x_min:x_max,:] = wordBB_gt

        return img, gt, wordBBs, texts



if __name__ == '__main__':
    base_dir = "/media/aerjay/Acer/Users/Aerjay/Downloads/datasets/icdar-2015"
    img_dir = os.path.join(base_dir, "4.1 - Text Localization/ch4_training_images")
    gt_dir = os.path.join(base_dir, "4.1 - Text Localization/ch4_training_localization_transcription_gt")
    # dataset = ICDAR2015Dataset(img_dir, gt_dir)
    dataset = PseudoGTDataset(img_dir, gt_dir)

    print(dataset[0])

    # img, wordBBs, texts = dataset[0]
    # print(f"img.shape = {img.shape}")

    # plt.figure()
    # plt.imshow(img)
    # plt.show()