import numpy as np
import cv2
import PIL
import random
import h5py
import os.path
from pathlib import Path
import matplotlib.pyplot as plt
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

import image_proc
import train_detector

class SynthCharDataset(Dataset):

    def __init__(self, gt_path, img_dir, size, batch_size_limit=64, augment=True):
        # inherit __init__() of Dataset class
        super(SynthCharDataset).__init__()

        self.gt_path = gt_path
        self.img_dir = img_dir

        self.size = size
        self.batch_size_limit = batch_size_limit
        self.augment = augment

        self.f = h5py.File(gt_path, 'r')
        self.length = len(self.f['imnames'])

        self.alphabet = list("AaBbCDdEeFfGgHhIiJjKLlMmNnOPQqRrSTtUVWXYZ") + [None,]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if changing starting index
        # idx += self.begin

        f = self.f

        # get the idx-th gt
        imgname = image_proc.u2ToStr(f[f['imnames'][idx][0]])
        charBBs = f[f['charBB'][idx][0]]
        wordBBs = f[f['wordBB'][idx][0]]
        txts = f[f['txt'][idx][0]]

        # get the idx-th image
        imgpath = os.path.join(self.img_dir, imgname)
        image = PIL.Image.open(imgpath)
        H,W = image.height, image.width

        # get chars (ground truth)
        chars = list("".join(image_proc.txtToInstance(txts)))

        # pepper in some nonchars
        if self.augment:
            charBBs, chars = BB_augment(charBBs, wordBBs, chars, (W,H),
                                batch_size_limit=self.batch_size_limit)

        # H,W = image.height, image.width
        C = 3   # channels
        N = len(chars)  # number of characters


        # get the cropped chars
        batch = np.zeros((N,C,*self.size))
        for i, charBB in enumerate(charBBs):
            if i >= self.batch_size_limit:
                break

            # crop + convert to numpy (H,W,C)
            cropped = cropBB(image, charBB, fast=True).astype('float32')

            # resize
            cropped = cv2.resize(cropped, dsize=self.size)  # numpy input

            # append to batch
            batch[i] = cropped.transpose(2,0,1)   # CHW
        image.close()

        # scale to [-1,1]
        batch /= 255.0  # [0,1]
        batch -= 0.5    # [-0.5,0.5]
        batch *= 2      # [-1,1]

        # convert to tensor
        batch = torch.from_numpy(batch).double()

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

def BB_augment(charBBs, wordBBs, gts, img_wh, augment=True, fraction_nonchar=0.1,
    batch_size_limit=64, expand_coeff=0.2, contract_coeff=0.2):
    N = min(len(gts), batch_size_limit) # batch size

    # filter faulty values
    skip_indices = []
    for i, charBB in enumerate(charBBs):
        tl, br = np.min(charBB, axis=0), np.max(charBB, axis=0)
        w, h = br - tl

        # if BB exceeds image bounds
        if br[0] >= img_wh[0] or br[1] >= img_wh[1]:
            skip_indices.append(i)
        # if BB dimensions <= 1
        elif w <= 1 or h <= 1:
            skip_indices.append(i)

    # turn h5py charBBs to np array
    charBBs_np = np.zeros((len(charBBs) - len(skip_indices),4,2))
    j = 0
    for i,charBB in enumerate(charBBs):
        if i not in skip_indices:
            charBBs_np[j] = charBB
            j += 1
    charBBs = charBBs_np

    # generate ``N_nonchars`` points not in wordBBs
    N_nonchars = int(fraction_nonchar*N)
    noncharBBs = genNonCharBBs(wordBBs, img_wh, N_nonchars, retries=10)

    # combine nonchars with chars
    if noncharBBs:
        charBBs = np.concatenate(charBBs, noncharBBs)
        gts += [None]*N_nonchars

    shuffle_in_unison(charBBs, gts)

    # limit the inputs to ``batch_size_limit``
    if len(gts) >= batch_size_limit:
        charBBs = charBBs[:N]
        gts = gts[:N]

    # perturb each coordinate in charBB, biased to increase the area
    for i,BB in enumerate(charBBs):
        if augment and gts[i]:  # augmentation
            # get longest side length
            c_BB = image_proc.get_containing_rect(BB)
            w_c, h_c = c_BB[2] - c_BB[0]

            low_x, low_y = -1*contract_coeff*w_c, -1*contract_coeff*h_c
            high_x, high_y = expand_coeff*w_c, expand_coeff*h_c
            noise_x = np.random.uniform(low=low_x, high=high_x, size=4)
            noise_y = np.random.uniform(low=low_y, high=high_y, size=4)
            noise = np.array([noise_x, noise_y]).T
            # t: -y     l: -x   b: +y   r: +x
            noise *= np.array([[-1,-1],
                               [ 1,-1],
                               [ 1, 1],
                               [-1, 1]])

            # ceil to prevent h=0 or w=0
            new_charBB = np.ceil(charBBs[i] + noise)
            new_charBB = np.clip(new_charBB + noise, 0, img_wh)

            # perturb only if dimensions > 1
            tl, br = np.min(BB, axis=0), np.max(BB, axis=0)
            new_width, new_height = br - tl
            if (new_width > 1) and (new_height > 1):
                charBBs[i] = new_charBB

    return charBBs, gts

def genNonCharBBs(wordBBs, img_wh, N_nonchars, retries=10):
    """
    # up to 5 failed retries
    # get random coordinate, determine its order (tl/tr/br/bl)
    # depending on its order, check if there is another coordinate
    # within ``nonchar_size`` of it. if there is, loop back.
    # if there is not, get the coordinates
    """
    # permutation =

    return None


def string_to_onehot(string, alphabet=None, char_to_int=None, include_nonchar=True,
                     to_onehot=True):
    # based on https://stackoverflow.com/questions/49370940/
    if not char_to_int:
        if not alphabet:
            alphabet = "AaBbCDdEeFfGgHhIiJjKLlMmNnOPQqRrSTtUVWXYZ"
        # a dict mapping every char of alphabet to unique int based on position
        char_to_int = dict((c,i) for i,c in enumerate(alphabet))
        if include_nonchar and (None not in char_to_int):
            char_to_int[None] = len(char_to_int)

    # convert string to array of ints
    encoded_data = []
    for char in string:
        if not isinstance(char, str):   # if non-character
            char = None
        else:
            if char not in alphabet:
                char = char.swapcase()
            if char not in alphabet:
                char = None     # if char not in alphabet

        encoded_data.append(char_to_int[char])
    # encoded_data = [char_to_int[char] if char in alphabet else char_to_int[char.swapcase()] for char in string]

    if to_onehot:
        # convert array of ints to one-hot vectors
        one_hots = torch.zeros(len(string), len(char_to_int))
        for i, j in enumerate(encoded_data):
            one_hots[i][j] = 1.0

        return one_hots
    else:
        return torch.Tensor(encoded_data)


def cropBB(img, BB, size=None, fast=False):
    # augment coordinates
    # BB = augmentBB(BB)

    # crop
    if fast:
        BB = image_proc.order_points(BB)
        j, i = np.min(BB, axis=0)
        # use ceil in w,h to prevent w=0 or h=0
        w, h = np.ceil(np.max(BB, axis=0) - np.min(BB, axis=0))

        cropped = image_proc.crop(img, i,j,h,w)
    else:
        if isinstance(img, PIL.Image.Image):
            img = np.array(img, dtype='float32')

        cropped = image_proc.perspectiveTransform(img, initial=BB, size=size)

    return cropped

def shuffle_in_unison(a, b):
    # source: https://stackoverflow.com/questions/4601373/
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def genBalancedCharDataset(N_max, img_dir, mat_path, char_dir, skip_existing=True,
                           classes=None):
    """ randomly select images forever
        for each selected image, save the character img
        and update the distribution
    """
    if not classes: # default classes:
        classes = list("AaBbCDdEeFfGgHhIiJjKLlMmNnOPQqRrSTtUVWXYZ")
    distribution = {c:0 for c in classes}
            # initialize distribution to the already-saved images

    f = h5py.File(mat_path, 'r')
    N_images = len(f['imnames'])
    # generate a random sequence of nonnegative integers < N_images
    for i in random.sample(range(N_images), N_images):
        if min(distribution.values()) >= N_max:
            break

        # determine the image number wrt matnum
        string = "".join(image_proc.txtToInstance(f[f['txt'][i][0]]))
        imname = image_proc.u2ToStr(f[f['imnames'][i][0]])
        img_path = os.path.join(img_dir, imname)
        # print(f"img_path: {img_path}") #

        for k, char in enumerate(string):
            if char not in distribution:
                continue
            if distribution[char] >= N_max:
                continue
            # determine output img path
            char_path = os.path.join(char_dir, char, f"{char}_{i}_{k}.png")
            print(char_path, end='')
            Path(os.path.split(char_path)[0]).mkdir(parents=True, exist_ok=True)
            if skip_existing and os.path.isfile(char_path):
                print("\t Skipping...")
                continue
            print("")

            # write img to path
            charBB = f[f['charBB'][i][0]][k]
            char_img = getCroppedImage(img_path, charBB)
            cv2.imwrite(char_path, char_img)

            distribution[char] += 1

    print(distribution)


def getCroppedImage(img_path, coords, augment=False):
    if augment:
        # perform augmentation
        return
    #else:
        #print("NO AUGMENTATION")
    if not os.path.isfile(img_path):
        print(f"WARNING: {img_path} DOES NOT EXIST. Skipping...")
        return None

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    cropped = image_proc.perspectiveTransform(img, initial=coords)
    return cropped


def synthetic_classifier_training():
    home = False
    if home:
        windows_path_prefix = "C:"
        linux_path_prefix = "/mnt/A4B04DFEB04DD806"

        path_prefix = linux_path_prefix
        img_dir = path_prefix + '/Users/Aerjay/Downloads/SynthText/SynthText'
        gt_path = path_prefix + '/Users/Aerjay/Downloads/SynthText/gt_v7.3.mat'
        weight_dir = '/home/aerjay/Documents/thesis/weights'
    else:
        gt_path = "/home/eee198/Downloads/SynthText/gt_v7.3.mat"
        img_dir = "/home/eee198/Downloads/SynthText/images"

        weight_folder = 'classifier/synth'
        weight_dir = "/home/eee198/Downloads/SynthText/weights/" + weight_folder
        # weight_fname = None     # pretrained weights

    # make weight_dir if it doesn't exist
    Path(weight_dir).mkdir(parents=True, exist_ok=True)

    cuda = False
    size = (64,64)

    epochs = range(1)

    dataset = SynthCharDataset(gt_path, img_dir, size)

    N = len(dataset)
    train_test_val = [int(0.8*N), int(0.15*N)]
    train_test_val += [N - sum(train_test_val),]
    train, test, validation = torch.utils.data.random_split(dataset, train_test_val)

    trainloader = DataLoader(train, batch_size=1, shuffle=True,
                                collate_fn=SynthCharDataset.collate_fn)
    # valloader = DataLoader(validation, batch_size=1, shuffle=True,
                                # collate_fn=SynthCharDataset.collate_fn)

    model = CharClassifier(num_classes=len(dataset.alphabet)).double()
    if cuda:
        model = model.cuda()

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #

    T_start = time.time()
    for epoch in epochs:
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0], data[1]
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss = train_detector.print_statistics(running_loss, loss,
                i, epoch, model=model, T_print=100, T_save=10000,
                T_start=T_start, weight_dir=weight_dir)
            running_loss += loss.item()

            # stopping criterion

    T_end = time.time()
    print('Finished Training')


class CharClassifier(nn.Module):

    def __init__(self, num_classes):
        super(CharClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(256 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        x = F.relu(self.conv6(x))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    pass
    # classes = ['a', 'e', 'i', 'o', 'u']
    # simpleGenChar(10, img_dir, mat_dir, N_images=100)
    # genBalancedCharDataset(20, img_dir, mat_path, char_dir, classes=classes)