import numpy as np
import cv2
import PIL
import h5py
import os.path
from pathlib import Path

import image_proc

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from craft.craft import CRAFT

import time


class SynthCharMapDataset(Dataset):
    """SynthText Dataset + Heatmap + Direction Ground Truths"""

    def __init__(self, gt_path, img_dir, begin=0, cuda=True, size=None,
                 color_flag=1, hard_examples=False, affinity_map=False,
                 character_map=True, word_map=False, direction_map=True):
        """
        Args:
            gt_path (string): Path to gt.mat file (GT file)
            img_dir (string): Path to directory of {i}/....jpg

            color_flag {1,0,-1}: Colored (1), Grayscale (0), or Unchanged (-1)
            begin (int): the index at which the Dataset is to begin
            cuda (bool): whether the output Tensors should be stored in GPU
        """
        # inherit __init__() of Dataset class
        super(SynthCharMapDataset).__init__()

        # paths
        self.gt_path = gt_path
        self.img_dir = img_dir

        # flags
        self.color_flag = color_flag
        self.hard_examples = hard_examples
        self.character_map = character_map
        self.affinity_map = affinity_map
        self.word_map = word_map
        self.direction_map = direction_map

        # templates
        if self.character_map:
            self.gaussian_template = image_proc.genGaussianTemplate()
        if self.direction_map:
            self.direction_template = image_proc.genDirectionMapTemplate()

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

        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if changing starting index
        idx += self.begin

        # extract the gt data from the mat-file
        f = self.f
        imgname = image_proc.u2ToStr(f[f['imnames'][idx][0]])
        charBBs = f[f['charBB'][idx][0]].value
        wordBBs = f[f['wordBB'][idx][0]].value
        txts    = f[f['txt'][idx][0]]

        # extract the idx-th image
        imgpath = os.path.join(self.img_dir, imgname)
        # read img, convert from HWC to CHW
        image = PIL.Image.open(imgpath)
        image_shape = image.height, image.width

        # get the heatmap GTs
        char_map, aff_map = image_proc.genPseudoGT(charBBs, txts, image_shape,
                                            generate_affinity=self.affinity_map,
                                            template=self.gaussian_template)
        if self.word_map:
            word_map = image_proc.genWordGT(wordBBs, image_shape,
                                            template=self.gaussian_template)
        if self.direction_map:
            cos_map, sin_map  = image_proc.genDirectionGT(charBBs, image_shape,
                                            template=self.direction_template)

        # combine gts into a single tensor
        gt = None
        if self.character_map:
            gt = char_map[None,...]
        if self.affinity_map:
            affinity_map = aff_map[None,...]
            if gt is None:
                gt = affinity_map
            else:
                gt = np.concatenate((gt, affinity_map))
        if self.word_map:
            word_map = word_map[None,...]
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

        # get hard examples + corresponding gts
        if self.hard_examples:
            hard_img, hard_gt = image_proc.hard_example_mining(image,
                                                               gt, wordBBs)
            # hard_img: NCHW
            # hard_gt: NCHW -> NHWC

            hard_gt = torch.from_numpy(hard_gt).type(self.dtype).permute(0,2,3,1)
            hard_img = torch.from_numpy(hard_img).type(self.dtype) / 255.0
        else:
            hard_img, hard_gt = None, None

        # transforms
        image, gt = image_proc.augment(image, gt, size=self.size, halve_gt=True)
        image = image.type(self.dtype) / 255.0 # CHW
        gt = gt.permute(1,2,0).type(self.dtype)    # resized, CHW->HWC

        if self.hard_examples:
            return image, gt, hard_img, hard_gt
        else:
            return image, gt


    def collate_fn(batch):
        imgs = [sample[0].permute(1,2,0) for sample in batch]   # CHW->HWC
        gts = [sample[1] for sample in batch]  # HWC

        h_img,w_img,_ = max_shape(imgs)
        h_gt,w_gt,_ = max_shape(gts)

        # if hard_examples
        hard_examples = (len(batch[0]) == 4)
        if hard_examples:
            hard_imgs = [sample[2].permute(1,2,0) for sample in batch]  # CHW->HWC
            hard_gts = [sample[3] for sample in batch]   # CHW->HWC

            h_himg,w_himg,_ = max_shape(hard_imgs)
            h_hgt,w_hgt,_ = max_shape(hard_gts)

        batch_resized = [[None]*len(batch[0]) for i in range(len(batch))]

        # resize
        for i in range(len(batch)):
            # images
            batch_resized[i][0] = torch.from_numpy(cv2.resize(imgs[i].numpy(),
                                    dsize=(w_img, h_img))).permute(2,0,1).cuda()
            # gts   (CHW -> HWC)
            batch_resized[i][1] = torch.from_numpy(cv2.resize(gts[i].numpy(),
                                    dsize=(w_gt, h_gt))).cuda()

            if hard_examples:
                batch_resized[i][2] = torch.from_numpy(cv2.resize(hard_imgs[i].numpy(),
                                        dsize=(w_himg, h_himg))).permute(2,0,1).cuda()
                batch_resized[i][3] = torch.from_numpy(cv2.resize(hard_gts[i].numpy(),
                                        dsize=(w_hgt, h_hgt))).cuda()

        return torch.utils.data._utils.collate.default_collate(batch_resized)

def max_shape(arr_tensors):
    shape = np.zeros((len(arr_tensors), arr_tensors[0].dim()))
    for i, tensor in enumerate(arr_tensors):
        shape[i] = tensor.shape

    return list(np.max(shape, axis=0).astype("int32"))


def show_samples(imgs, i=None, feature_type="img", title=None, channel=None):
    imgs = imgs.detach().cpu().numpy()
    if i == None:
        img = imgs
    else:
        img = imgs[i]

    if feature_type == "img":
        img = img.transpose(1,2,0)
    elif feature_type == "gt":
        pass
    else:
        print(f"Warning: feature_type should be 'img' or 'gt', not " +
              f" '{feature_type}'. Assuming 'gt'.")

    if channel != None:
        img = img[:,:,channel]

    if title is None:
        title = f"img[{i}]"

    plt.figure(figsize=(10,10))
    plt.title(title)
    plt.imshow(img, interpolation='nearest')

    plt.show()


def print_statistics(running_loss, loss, i, epoch, model=None,
                     T_start=None, T_print=100, T_save=10000, weight_dir=None,
                     weight_fname_template=None, weight_fname_args=None,
                     print_template=None, print_args=None):
    running_loss += loss.item()

    # print every T_print minibatches
    if i % T_print == T_print-1:
        if print_template is None:
            print_template = '[%d, %5d] loss: %f'
            print_args = (epoch + 1, i + 1, running_loss/T_print)
        print(print_template % print_args)
        running_loss = 0.0

    # save every T_save minibatches
    if i % T_save == T_save-1:
        if not (weight_dir is None):
            if weight_fname_template is None:
                weight_fname_template = "w_%d.pth"
                weight_fname_args = (i+1,)

            weight_fname = weight_fname_template % weight_fname_args
            weight_path = os.path.join(weight_dir, weight_fname)

            print(f"\nsaving at {i+1}-th batch")
            torch.save(model.state_dict(), weight_path)
            T_end = time.time()
            print(f"\nElapsed time: {T_end-T_start}")

    return running_loss

def step(model, criterion, optimizer, input, target):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    output,_ = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    return loss, optimizer

def init_data(gt_path, img_dir, dataset_kwargs={}, dataloader_kwargs={}, **kwargs):
    # defaults:
    dataset_defaults = {}
    dataloader_defaults = {
        "batch_size": 4,
        "shuffle": True
    }
    # if size is set, no need to use custom collate_fn
    if ("size" not in dataset_kwargs) or (dataset_kwargs["size"] is None):
        dataloader_defaults["collate_fn"] = SynthCharMapDataset.collate_fn

    for entry in dataset_defaults:
        if entry not in dataset_kwargs:
            dataset_kwargs[entry] = dataset_defaults[entry]
    for entry in dataloader_defaults:
        if entry not in dataloader_kwargs:
            dataloader_kwargs[entry] = dataloader_defaults[entry]
    if "split" not in kwargs:
        kwargs["split"] = [800000,58750]

    # remember requires_grad=True
    dataset = SynthCharMapDataset(gt_path, img_dir, **dataset_kwargs)
    train, test = torch.utils.data.random_split(dataset, kwargs["split"])
    dataloader = DataLoader(train, **dataloader_kwargs)

    return dataloader, train, test

def init_model(weight_dir, weight_fname=None, num_class=3):
    # make weight_dir if it doesn't exist
    Path(weight_dir).mkdir(parents=True, exist_ok=True)

    # input: NCHW
    model = CRAFT(pretrained=True, num_class=num_class).cuda()
    # output: NHWC

    if weight_fname:
        pretrained_weight_path = os.path.join(weight_dir, weight_fname)
        model.load_state_dict(torch.load(pretrained_weight_path))
        model.eval()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001) # tweak parameters

    return model, criterion, optimizer

def train_loop(dataloader, model, criterion, optimizer, weight_dir, epochs=1):
    T_start = time.time()
    for epoch in range(epochs):
        running_loss, hard_running_loss = 0.0, 0.0

        while True:
            try:
                for i, data in enumerate(dataloader):
                    # print(i, end='')

                    # unpack data:
                    if len(data) == 2:
                        img, target = data
                    else:
                        img, target, hard_img, hard_target = data

                    loss, optimizer = step(model, criterion, optimizer, img, target)
                    running_loss = print_statistics(running_loss, loss, i,
                                        epoch, model=model, T_start=T_start,
                                        weight_dir=weight_dir)

                    if len(data) == 2:
                        continue
                    hard_img, hard_target = hard_img[0], hard_target[0]

                    loss, optimizer = step(model, criterion, optimizer, hard_img, hard_target)
                    hard_running_loss = print_statistics(hard_running_loss, loss,
                                                i, epoch, T_start=T_start)
                break
            except KeyboardInterrupt:
                weight_fname_interrupt = f"w_{i+1}_interrupt.pth"
                weight_path_interrupt = os.path.join(weight_dir, weight_fname_interrupt)

                print(f"\nSaving at {i+1}-th batch...")
                torch.save(model.state_dict(), weight_path_interrupt)

                T_end = time.time()
                print(f"Elapsed time: {T_end-T_start}")
                break
            except Exception as err:
                print("")
                print(err)
            else:
                print(f"Occured at i={i}, ")
                print(f"img.shape = {img.shape}, target.shape = {target.shape}")
                print(f"hard_img.shape = {hard_img.shape}, hard_target.shape ="
                        + f"{hard_target.shape}")

    print("Finished training.")

    T_end = time.time()
    print(f"\nTotal elapsed time: {T_end-T_start}")

    return img, target

def main(weight_folder):
    gt_path = "/home/eee198/Downloads/SynthText/gt_v7.3.mat"
    img_dir = "/home/eee198/Downloads/SynthText/images"
    weight_dir = "/home/eee198/Downloads/SynthText/weights/" + weight_folder
    weight_fname = None     # pretrained weights

    dataset_kwargs = {
        "cuda": True
    }
    dataloader_kwargs = {
        "batch_size": 4
    }
    num_class = 3
    epochs = 1

    dataloader, train, test = init_data(gt_path, img_dir,
            dataset_kwargs=dataset_kwargs, dataloader_kwargs=dataloader_kwargs)
    model, criterion, optimizer = init_model(weight_dir, weight_fname, num_class)

    train_loop(dataloader, model, criterion, optimizer, weight_dir, epochs=epochs)


if __name__ == '__main__':
    main()
