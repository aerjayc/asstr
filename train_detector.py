import numpy as np
import cv2
import h5py
import os.path

import image_proc
import gt_io

import torch
import torchvision
from torch.utils.data import Dataset

from craft.craft import CRAFT


class SynthCharMapDataset(Dataset):
    """SynthText Dataset + Heatmap + Direction Ground Truths"""

    def __init__(self, gt_dir, root_dir, color_flag=1, transform=None):
        """
        Args:
            gt_dir (string): Path to gt_{i}.mat files (GT files)
            root_dir (string): Path to {j}/....jpg (folders of images)

            color_flag {1,0,-1}: Colored (1), Grayscale (0), or Unchanged (-1)
        """
        super(SynthCharMapDataset).__init__()

        self.gt_dir = gt_dir
        self.root_dir = root_dir
        self.color_flag = color_flag
        self.transform = transform

        self.length = 0
        fs = []
        for filename in os.scandir(gt_dir):
            if filename.endswith('.mat'):
                f = h5py.File(filename, 'r')
                fs.append(f)
                self.length += len(f['imnames'])
        self.fs = fs

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        bound = 50000
        matnum = int(idx/bound)
        imgnum = idx % bound

        f = self.fs[matnum]
        imgname = f[f['imnames'][imgnum][0]]
        charBBs = f[f['charBB'][imgnum][0]]
        txts    = f[f['txt'][imgnum][0]]

        imgpath = os.path.join(self.root_dir, imgname)
        synthetic_image = cv2.imread(imgpath, self.color_flag)
        image_shape = synthetic_image.shape[0:2]

        char_map, aff_map = image_proc.genPseudoGT(charBBs, txts, image_shape)
        cos_map, sin_map  = image_proc.genDirectionGT(charBBs, image_shape)

        # gt = {'char_map': char_map, 'aff_map': aff_map,
        #       'cos_map': cos_map, 'sin_map': sin_map}

        gt = np.concatenate(char_map, aff_map, cos_map, sin_map)

        height, width = image_shape
        gt_shape = int(width/2), int(height/2)
        for j, img in enumerate(gt):
            gt[j,:,:] = cv2.resize(img, gt_shape, interpolation=cv2.INTER_AREA)

        return synthetic_image, gt


if __name__ == '__main__':
    model = CRAFT(pretrained=False)
    output, features = model(torch.randn(1,3,768,768))
    print(output.shape)