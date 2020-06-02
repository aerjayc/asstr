import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import cv2
import PIL.Image
import scipy.ndimage
import image_proc

import h5py
import pandas as pd

import os
import os.path

from dataset_utils import shuffle_in_unison, max_shape, h5py_to_numpy,\
                          u2ToStr, txtToInstance, string_to_onehot,\
                          filter_BBs, BB_augment
from image_proc import cropBB


ALPHABET = list("AaBbCDdEeFfGgHhIiJjKLlMmNnOPQqRrSTtUVWXYZ") + [None,]

# SynthText Dataset
class SynthCharDataset(Dataset):

    def __init__(self, gt_path, img_dir, size, batch_size_limit=64,
                 augment=True, shuffle=True, normalize=True):
        # inherit __init__() of Dataset class
        super(SynthCharDataset).__init__()

        self.gt_path = gt_path
        self.img_dir = img_dir

        self.size = size
        self.batch_size_limit = batch_size_limit
        self.augment = augment
        self.shuffle = shuffle
        self.normalize = normalize

        self.f = h5py.File(gt_path, 'r')
        self.length = len(self.f['imnames'])

        self.alphabet = ALPHABET

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if changing starting index
        # idx += self.begin

        f = self.f

        # get the idx-th gt
        imgname = u2ToStr(f[f['imnames'][idx][0]])
        charBBs = f[f['charBB'][idx][0]]
        wordBBs = f[f['wordBB'][idx][0]]
        txts = f[f['txt'][idx][0]]

        # get the idx-th image
        imgpath = os.path.join(self.img_dir, imgname)
        image = PIL.Image.open(imgpath)
        W,H = image.width, image.height

        # get chars (ground truth)
        chars = list("".join(txtToInstance(txts)))


        # convert charBBs (which is h5py) to numpy
        charBBs = h5py_to_numpy(charBBs)

        # filter faulty values
        charBBs, chars = filter_BBs(charBBs, chars, (W,H))

        # pepper in some nonchars
        if self.augment:
            # Note: BB_augment is forced to shuffle_in_unison if nonchar=True
            charBBs, chars = BB_augment(charBBs, wordBBs, chars, (W,H),
                                batch_size_limit=self.batch_size_limit,
                                shuffle=self.shuffle)
        else:
            if self.shuffle:
                shuffle_in_unison(charBBs, chars)

            charBBs = charBBs[:self.batch_size_limit]
            chars = chars[:self.batch_size_limit]

        C = 3   # channels
        N = len(chars)  # number of characters

        # get the cropped chars
        batch = np.zeros((N,C,*self.size))
        for i, charBB in enumerate(charBBs):
            # crop + convert to numpy (H,W,C)
            cropped = cropBB(image, charBB, fast=True).astype('float32')

            # resize
            cropped = cv2.resize(cropped, dsize=self.size)  # numpy input

            # augment further
            if self.augment:
                angle = transforms.RandomRotation.get_params([-45, 45])
                cropped = scipy.ndimage.rotate(cropped, angle, reshape=False)

            # append to batch
            batch[i] = cropped.transpose(2,0,1)   # CHW
        image.close()

        if self.normalize:
            # scale to [-1,1]
            batch /= 255.0  # [0,1]
            batch -= 0.5    # [-0.5,0.5]
            batch *= 2      # [-1,1]

        # convert to tensor
        batch = torch.from_numpy(batch)#.double()

        # convert chars to stack of one-hot vectors
        onehot_chars = string_to_onehot(chars, alphabet=self.alphabet,
                            to_onehot=False).long()

        return batch, onehot_chars


    def collate_fn(batch):
        imgs = None
        for sample in batch:
            if not imgs:
                imgs = sample[0]
                chars = sample[1]
            else:
                imgs = torch.cat((imgs, sample[0]))
                chars = torch.cat((chars, sample[1]))

        return imgs, chars


class SynthCharMapDataset(Dataset):
    """SynthText Dataset + Heatmap + Direction Ground Truths"""

    def __init__(self, gt_path, img_dir, begin=0, cuda=True, size=None,
                 color_flag=1, hard_examples=False, affinity_map=True,
                 character_map=True, word_map=False, direction_map=False):
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
        imgname = u2ToStr(f[f['imnames'][idx][0]])
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


class ICDAR2013Dataset(Dataset):

    def __init__(self, gt_dir, img_dir):
        # inherit  __init__() of Dataset class
        super(ICDAR2013Dataset).__init__()

        self.gt_dir = gt_dir
        self.img_dir = img_dir

        img_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        self.img_names = []
        # self.img_paths = []
        _, _, files = next(os.walk(img_dir))
        for file in sorted(files):
            if os.path.splitext(file)[1].lower() in img_exts:
                self.img_names.append(file)
                # path_name = os.path.join(img_dir, file)
                # self.img_paths.append(path_name)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_names[idx]
        gt_name = f"{img_name[:3]}_GT.txt"  # generalize this
        img_path = os.path.join(self.img_dir, img_name)
        gt_path = os.path.join(self.gt_dir, gt_name)

        img = PIL.Image.open(img_path)

        headers = ['R', 'G' ,'B' ,  # RGB values
                   'x0', 'y0',      # center
                   'x1', 'y1',      # top left
                   'x2', 'y2',      # bottom right
                   'character']
        gt_df = pd.read_csv(gt_path,
                            names=headers,
                            comment='#',
                            skip_blank_lines=False, # to separate words
                            delim_whitespace=True,
                            doublequote=False)      # to get """ entries

        txt = gt_df['character'].copy()
        txt[txt.isnull()] = ' '
        txt = list(txt)

        gt_df = gt_df.dropna()              # drop NaN rows

        charBBs = gt_df[['x1', 'y1',
                         'x2', 'y1',
                         'x2', 'y2',
                         'x1', 'y2']].to_numpy().reshape(-1,4,2)

        chars = gt_df['character'].to_numpy()

        return img, charBBs, chars, txt


class ICDAR2013TestDataset(Dataset):

    def __init__(self, gt_dir, img_dir, size=None, cuda=True, normalize=True):
        # inherit  __init__() of Dataset class
        super(Dataset).__init__()

        self.gt_dir = gt_dir
        self.img_dir = img_dir
        self.size = size
        self.cuda = cuda
        self.normalize = normalize

        img_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        self.img_names = []
        # self.img_paths = []
        _, _, files = next(os.walk(img_dir))
        for file in sorted(files):
            if os.path.splitext(file)[1].lower() in img_exts:
                self.img_names.append(file)
                # path_name = os.path.join(img_dir, file)
                # self.img_paths.append(path_name)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_names[idx]
        gt_name = f"gt_{img_name[:-4]}.txt"
        img_path = os.path.join(self.img_dir, img_name)
        gt_path = os.path.join(self.gt_dir, gt_name)

        pil_img = PIL.Image.open(img_path)
        W, H = pil_img.width, pil_img.height
        img = np.array(pil_img)             # H,W,C
        pil_img.close()

        headers = ['left', 'top', 'right', 'bottom', 'transcription']
        gt_df = pd.read_csv(gt_path,
                            names=headers,
                            comment='#',
                            delim_whitespace=True,
                            escapechar='\\')

        wordBBs = gt_df[['left' , 'top'   ,
                         'right', 'top'   ,
                         'right', 'bottom',
                         'left' , 'bottom' ]].to_numpy().reshape(-1,4,2)

        words = gt_df['transcription'].to_numpy()

        # resize when # of pixels exceeds size
        if (self.size is not None) and ((W*H) > np.multiply(*self.size)):
            img = cv2.resize(img, dsize=self.size)

        # normalize to [-1,1]
        if self.normalize:
            img = img.astype('float')
            img /= 255.0
            img -= 0.5
            img *= 2.0

        img = torch.from_numpy(img).permute(2,0,1).float()  # C,H,W
        if self.cuda:
            img = img.cuda()

        return img, wordBBs, words


class ICDAR2013MapDataset(ICDAR2013Dataset):

    def __init__(self, gt_dir, img_dir, size=None, cuda=True, augment=True,
                 character_map=True, affinity_map=True, direction_map=False):
        self.raw_dataset = ICDAR2013Dataset(gt_dir, img_dir)
        self.character_map = character_map
        self.affinity_map = affinity_map
        self.direction_map = direction_map
        self.augment = augment

        self.size = size
        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        # generate templates
        if self.character_map:
            self.gaussian_template = image_proc.genGaussianTemplate()
        if self.direction_map:
            self.direction_template = image_proc.genDirectionMapTemplate()

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, charBBs, chars, txt = self.raw_dataset[idx]
        img_shape = img.height, img.width

        # get the heatmap GTs
        if self.character_map:
            char_map, aff_map = image_proc.genPseudoGT(charBBs, txt, img_shape,
                                        generate_affinity=self.affinity_map,
                                        template=self.gaussian_template)
        if self.direction_map:
            cos_map, sin_map = image_proc.genDirectionGT(charBBs, img_shape,
                                        template=self.direction_template)

        # combine gts into a single tensor
        gt = None
        if self.character_map:
            gt = char_map[None, ...]
        if self.affinity_map:
            if gt is None:
                gt = aff_map[None, ...]
            else:
                gt = np.concatenate((gt, aff_map[None, ...]))
        if self.direction_map:
            dir_map = np.stack((cos_map, sin_map))
            if gt is None:
                gt = dir_map
            else:
                gt = np.concatenate((gt, dir_map))

        if self.augment:
            img, gt = image_proc.augment(img, gt, size=self.size, halve_gt=True)
        else:
            # convert to torch tensor
            img = torch.Tensor(np.array(img)).permute(2,0,1)  # CHW
            gt = torch.Tensor(gt)                   # CHW
            gt = F.interpolate(gt[None,...], scale_factor=0.5)[0]
            # img, gt = image_proc.augment(img, gt, size=self.size, halve_gt=True,
                            # scale=(1.0,1.0),ratio=(1,1),degrees=[0,0])
        img = img.type(self.dtype) / 255.0  # CHW
        gt = gt.permute(1,2,0).type(self.dtype) # resized, CHW->HWC

        return img, gt


class ICDAR2013CharDataset(ICDAR2013Dataset):

    def __init__(self, gt_dir, img_dir, augment=True, size=(64,64),
                 batch_size_limit=None, shuffle=True, normalize=True):
        self.raw_dataset = ICDAR2013Dataset(gt_dir, img_dir)

        self.augment = augment
        self.size = size

        self.batch_size_limit = batch_size_limit
        self.shuffle = shuffle
        self.normalize = normalize

        self.alphabet = ALPHABET

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, charBBs, chars, txt = self.raw_dataset[idx]
        W, H = img.width, img.height

        # filter faulty values
        charBBs, chars = filter_BBs(charBBs, chars, (W,H))

        # pepper in some nonchars
        if self.augment:
            charBBs, chars = BB_augment(charBBs, None, chars, (W,H),
                                contract_coeff=0.1, expand_coeff=0.3,
                                batch_size_limit=self.batch_size_limit,
                                shuffle=self.shuffle)
        else:
            if self.shuffle:
                shuffle_in_unison(charBBs, chars)

            charBBs = charBBs[:self.batch_size_limit]
            chars = chars[:self.batch_size_limit]

        C = 3   # channels
        N = len(chars)

        # get the cropped chars
        batch = np.zeros((N,C,*self.size))
        for i, charBB in enumerate(charBBs):
            # crop + convert to numpy (H,W,C)
            cropped = cropBB(img, charBB, fast=False).astype('float32')

            # resize
            cropped = cv2.resize(cropped, dsize=self.size)

            # augment further
            if self.augment:
                angle = transforms.RandomRotation.get_params([-45, 45])
                cropped = scipy.ndimage.rotate(cropped, angle, reshape=False)

            # append to batch
            batch[i] = cropped.transpose(2,0,1) # CHW
        img.close()

        if self.normalize:
            # scale to [-1,1]
            batch /= 255.0  # [0,1]
            batch -= 0.5    # [-0.5,0.5]
            batch *= 2      # [-1,1]

        # convert to tensor
        batch = torch.from_numpy(batch)

        # convert chars to stack of one-hot vectors
        onehot_chars = string_to_onehot(chars, alphabet=self.alphabet,
                                    to_onehot=False).long()

        return batch, onehot_chars

