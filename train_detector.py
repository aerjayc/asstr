import numpy as np
import cv2
import h5py
import os.path

import image_proc
import gt_io

import torch
import torchvision
from torch import nn
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

    def __init__(self, gt_dir, root_dir, color_flag=1, #transform=None,
                 character_map=True, affinity_map=True, direction_map=True):
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
        #self.transform = transform
        #self.transform = transforms.Normalize((0,0,0), (1,1,1))
        self.character_map = character_map
        self.affinity_map = affinity_map
        self.direction_map = direction_map

        self.length = 0
        fs = []
        for filename in os.scandir(gt_dir): # not sorted
            if filename.name.endswith('.mat'):
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
        imgname = image_proc.u2ToStr(f[f['imnames'][imgnum][0]])
        charBBs = f[f['charBB'][imgnum][0]]
        txts    = f[f['txt'][imgnum][0]]

        imgpath = os.path.join(self.root_dir, imgname)
        synthetic_image = cv2.imread(imgpath, self.color_flag)
        synthetic_image = torch.from_numpy(synthetic_image).float().cuda().permute(2,0,1)   # CHW
        image_shape = synthetic_image.shape[1:]

        char_map, aff_map = image_proc.genPseudoGT(charBBs, txts, image_shape,
                                                   generate_affinity=self.affinity_map)
        if self.direction_map:
            cos_map, sin_map  = image_proc.genDirectionGT(charBBs, image_shape)

        # stack the maps
        gt = None
        if self.character_map:
            gt = extendDim(char_map)
        if self.affinity_map:
            affinity_map = extendDim(aff_map)
            if type(gt) == type(None):
                gt = affinity_map
            else:
                gt = np.concatenate((gt, affinity_map))
        if self.direction_map:
            dir_map = np.concatenate((cos_map, sin_map))
            if type(gt) == type(None):
                gt = dir_map
            else:
                gt = np.concatenate(gt, dir_map)

        # resize to match feature map size
        height, width = image_shape
        gt_shape = int(width/2), int(height/2)
        for j, img in enumerate(gt):    # inefficient
            map_resized = cv2.resize(img, (gt_shape[1],gt_shape[0]), interpolation=cv2.INTER_AREA)
            if j == 0:
                gt_resized = extendDim(map_resized)
            else:
                gt_resized = np.concatenate(gt_resized, map_resized)

        gt_resized = torch.from_numpy(gt_resized).float().cuda()    # CHW
        gt_resized = gt_resized.permute(1,2,0)  # HWC

        # normalization
        synthetic_image = synthetic_image / 255.0   # turn this into a transform

        return synthetic_image, gt


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