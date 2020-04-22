import numpy as np
import pandas
import os
import cv2
import PIL

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from craft.craft import CRAFT
import torch.nn.functional as F
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
        img = PIL.Image.open(img_path)  # cv2.imread(img_path, self.color_flag)

        gt_name = f"gt_img_{idx}.txt"
        gt_path = os.path.join(self.gt_dir, gt_name)

        headers = ['x1','y1','x2','y2','x3','y3','x4','y4','transcription']
        gt_df  = pandas.read_csv(gt_path, names=headers)
        wordBBs = gt_df[['x1','y1','x2','y2','x3','y3','x4','y4']].to_numpy().reshape(-1,4,2)
        texts = gt_df['transcription']

        return img, wordBBs, texts

class PseudoGTDataset(ICDAR2015Dataset):
    def __init__(self, img_dir, gt_dir, weights_path=None, color_flag=1,
                 character_map=True, affinity_map=False, word_map=False,
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

        # C,H,W = img.shape
        H,W = img.height, img.width
        gt = np.zeros((H // 2, W // 2, self.num_class), dtype="float32")
        for wordBB in wordBBs:
            x_min, y_min = np.min(wordBB, axis=0).astype("int32")
            x_max, y_max = np.max(wordBB, axis=0).astype("int32")
            wordBB_img = img.crop((x_min, y_min, x_max, y_max)) # img[:,y_min:y_max, x_min:x_max]
            wordBB_img = transforms.functional.to_tensor(wordBB_img)    # CHW
            # print(f"wordBB_img.shape = {wordBB_img.shape}")
            # wordBB_img.show()
            # input("press Enter to continue...")

            wordBB_gt,_ = model(wordBB_img)
            # wordBB_gt = F.interpolate(wordBB_img[None,...], scale_factor=0.5)[0]\
            #           .permute(1,2,0)   # HWC

            y_min, x_min = y_min // 2, x_min // 2
            y_max = y_min + wordBB_gt.shape[0]
            x_max = x_min + wordBB_gt.shape[1]
            gt[y_min:y_max,x_min:x_max,:] = wordBB_gt

        return img, gt, wordBBs, texts

def genConfidence(charBBs, gt_shape, p_charBBs):
    s = torch.ones(gt_shape).astype("float32")

    # conditional pdfs of each charBB
    #p_charBBs = [[], [], ...]
    # preprocess to get scalar values (probability of being a char)
    #conf_charBBs = []

    # tile all charBBs with their corresponding confidences
    for charBB, conf_charBB in zip(charBBs, conf_charBBs):
        # assumes charBB is oriented at angle 0
        x_min, y_min = torch.min(charBB, axis=0).astype("int32")
        x_max, y_max = torch.max(charBB, axis=0).astype("int32")

        charBB_shape = gt_shape[0], (y_max - y_min), (x_max - x_min)
        # s[charBB] = conf_charBB*torch.ones(shape of charBB)
        s[:,y_min:y_max,x_min:x_max] = conf_charBB*torch.ones(charBB_shape)

    return s


if __name__ == '__main__':
    base_dir = "/media/aerjay/Acer/Users/Aerjay/Downloads/datasets/icdar-2015"
    img_dir = os.path.join(base_dir, "4.1 - Text Localization/ch4_training_images")
    gt_dir = os.path.join(base_dir, "4.1 - Text Localization/ch4_training_localization_transcription_gt")
    dataset = PseudoGTDataset(img_dir, gt_dir)

    img, gt, wordBBs, texts = dataset[0]
    print(f"gt.shape = {gt.shape}")

    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(gt)
    plt.show()