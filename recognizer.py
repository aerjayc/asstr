import numpy as np
import torch

import cv2
import PIL.Image

from image_proc import genCharBB, cropBB
from dataset_utils import filter_BBs, expand_BBs

def map_to_crop(img, char_map, char_size=(64,64), cuda=True, expand_factor=2.5):
    if isinstance(img, PIL.Image.Image):
        C, W, H = 3, img.width, img.height
    if isinstance(img, (np.ndarray, torch.Tensor)):
        if img.shape[0] == 3:
            C, H, W = img.shape
            if isinstance(img, np.ndarray):
                img = img.transpose(1,2,0)
            else:   # if torch.Tensor
                img = img.permute(1,2,0).cpu().detach().numpy()
        else:
            H, W, C = img.shape

    if isinstance(char_map, torch.Tensor):
        char_map = char_map.cpu().detach().numpy()

    # generate character bounding boxes from heatmap
    charBBs = genCharBB(char_map, xywh=False) * 2

    # expand
    charBBs = expand_BBs(charBBs, (W,H), factor=expand_factor)

    # filter faulty values
    charBBs, _ = filter_BBs(charBBs, None, (W,H), verbose=True)

    # get the cropped chars
    N = len(charBBs)
    cropped_chars = np.zeros((N, C, *char_size))
    for i, charBB in enumerate(charBBs):
        # crop + convert to numpy (H, W, C)
        cropped = cropBB(img, charBB, fast=True).astype('float32')

        # resize
        cropped = cv2.resize(cropped, dsize=char_size)

        cropped_chars[i] = cropped.transpose(2,0,1) # CHW

    if isinstance(img, PIL.Image.Image):
        img.close()

    cropped_chars = torch.from_numpy(cropped_chars)

    if cuda:
        cropped_chars = cropped_chars.cuda()

    return cropped_chars, charBBs
