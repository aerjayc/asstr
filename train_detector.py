import numpy as np
import cv2
import h5py
import os.path

import image_proc
import gt_io

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from craft.craft import CRAFT


def extendDim(tensor):
    if torch.is_tensor(tensor):
        return tensor.view((1,) + tensor.size())
    else:
        return tensor.reshape((1,) + tensor.shape)

class SynthCharMapDataset(Dataset):
    """SynthText Dataset + Heatmap + Direction Ground Truths"""

    def __init__(self, gt_path, img_dir, color_flag=1, begin=0, cuda=True, #transform=None,
                 character_map=True, affinity_map=True, word_map=True, direction_map=True):
        """
        Args:
            gt_path (string): Path to gt.mat file (GT file)
            img_dir (string): Path to directory of {i}/....jpg (folders of images)

            color_flag {1,0,-1}: Colored (1), Grayscale (0), or Unchanged (-1)
        """
        super(SynthCharMapDataset).__init__()

        # paths
        self.gt_path = gt_path
        self.img_dir = img_dir

        # flags
        self.color_flag = color_flag
        self.character_map = character_map
        self.affinity_map = affinity_map
        self.word_map = word_map
        self.direction_map = direction_map

        self.f = h5py.File(gt_path, 'r')
        self.length = len(self.f['imnames'])

        self.begin = 0
        if begin > 0:
            self.begin = begin
            self.length -= begin

        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if changing starting index
        idx += self.begin

        f = self.f
        imgname = image_proc.u2ToStr(f[f['imnames'][idx][0]])
        charBBs = f[f['charBB'][idx][0]]
        wordBBs = f[f['wordBB'][idx][0]]
        txts    = f[f['txt'][idx][0]]

        imgpath = os.path.join(self.img_dir, imgname)
        synthetic_image = cv2.imread(imgpath, self.color_flag)# HWC
        image_shape = synthetic_image.shape[0:2]
        synthetic_image = torch.from_numpy(synthetic_image).type(self.dtype).permute(2,0,1)# CHW


        char_map, aff_map = image_proc.genPseudoGT(charBBs, txts, image_shape,
                                                   generate_affinity=self.affinity_map)
        if self.word_map:
            word_map = image_proc.genWordGT(wordBBs, image_shape)
        if self.direction_map:
            cos_map, sin_map  = image_proc.genDirectionGT(charBBs, image_shape)


        gt = None
        if self.character_map:
            gt = extendDim(char_map)
        if self.affinity_map:
            affinity_map = extendDim(aff_map)
            if gt is None:
                gt = affinity_map
            else:
                gt = np.concatenate((gt, affinity_map))
        if self.word_map:
            word_map = extendDim(word_map)
            if gt is None:
                gt = word_map
            else:
                gt = np.concatenate((gt, word_map))
        if self.direction_map:
            dir_map = np.stack((cos_map, sin_map))
            if gt is None:
                gt = dir_map
            else:
                gt = np.concatenate((gt, dir_map))


        # resize to match feature map size
        # to match expectations of F.interpolate, we reshape to NCHW
        gt = extendDim(torch.from_numpy(gt).type(self.dtype))
        gt_resized = F.interpolate(gt, scale_factor=0.5)[0].permute(1,2,0)# HWC

        #synthetic_image = self.transform(synthetic_image)
        synthetic_image = synthetic_image / 255.0

        return synthetic_image.astype('float32'), gt_resized.astype('float32')


if __name__ == '__main__':
    # model = CRAFT(pretrained=True)
    # output, features = model(torch.randn(1,3,768,768))
    # print(output.shape)

    gt_dir = "/home/eee198/Downloads/SynthText/matfiles"
    img_dir = "/home/eee198/Downloads/SynthText/images"


    # remember requires_grad=True
    dataset = SynthCharMapDataset(gt_dir, img_dir, affinity_map=False, direction_map=False,
                                  begin=10000)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # input: NCHW
    # output: NHWC
    model = CRAFT(pretrained=True, num_class=1).cuda()

    # weight_path = "/home/eee198/Downloads/SynthText/weights/w_10000"
    # model.load_state_dict(torch.load(weight_path))
    # model.eval()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001) # tweak parameters

    T_save = 10000
    T = 100
    epochs = 1
    start = time.time()
    for epoch in range(epochs):
        running_loss = 0.0

        for i, (img, target) in enumerate(dataloader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output, _ = model(img)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % T == T-1:    # print every T mini-batches
                print('[%d, %5d] loss: %f' % (epoch + 1, i + 1, running_loss/T))
                running_loss = 0.0
            if i % T_save == T_save-1:
                print(f"\nsaving at {i}-th batch'\n")
                torch.save(model.state_dict(), f"/home/eee198/Downloads/SynthText/weights/w_{i}.pth")
                end = time.time()
                print(f"\nElapsed time: {end-start}")

    print("Finished training.")

    end = time.time()
    print(f"\nTotal elapsed time: {end-start}")