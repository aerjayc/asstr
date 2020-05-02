import numpy as np
import cv2
import PIL
import random
import h5py
import os.path
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

import image_proc

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
            charBBs, chars = BB_augment(charBBs, wordBBs, chars, (W,H))

        # H,W = image.height, image.width
        C = 3   # channels
        N = len(chars)  # number of characters
        # if N > self.batch_size_limit:
            # N = self.batch_size_limit
            # chars = chars[:N]


        # get the cropped chars
        batch = np.zeros((N,C,*self.size))
        for i, charBB in enumerate(charBBs):
            if i >= self.batch_size_limit:
                break

            # crop + convert to numpy (H,W,C)
            cropped = cropBB(image, charBB, fast=False).astype('float32')

            # resize
            cropped = cv2.resize(cropped, dsize=self.size)  # numpy input

            if i == 10:
                print(chars[i])
                plt.imshow(cropped.astype('int32'))
                plt.show()

            # append to batch
            batch[i] = cropped.transpose(2,0,1)   # CHW
        image.close()


        # convert to tensor
        batch = torch.from_numpy(batch)

        # convert chars to stack of one-hot vectors
        onehot_chars = string_to_onehot(chars)

        return cropped, onehot_chars


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

def BB_augment(charBBs, wordBBs, gts, img_wh, fraction_nonchar=0.1,
    batch_size_limit=64, expand_coeff=0.3, contract_coeff=0.1):
    N = min(len(gts), batch_size_limit) # batch size

    # turn h5py charBBs to np array
    charBBs_np = np.zeros((len(charBBs),4,2))
    for i,charBB in enumerate(charBBs):
        charBBs_np[i] = charBB
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
        BBs = BBs[:N]
        gts = gts[:N]

    # perturb each coordinate in charBB, biased to increase the area
    for i,BB in enumerate(charBBs):
        if gts[i]:  # augmentation
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

            charBBs[i] += noise
            charBBs[i] = np.clip(charBBs[i] + noise, 0, img_wh)

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


def string_to_onehot(string, alphabet=None, char_to_int=None):
    # based on https://stackoverflow.com/questions/49370940/
    if not char_to_int:
        if not alphabet:
            alphabet = "AaBbCDdEeFfGgHhIiJjKLlMmNnOPQqRrSTtUVWXYZ"
        # a dict mapping every char of alphabet to unique int based on position
        char_to_int = dict((c,i) for i,c in enumerate(alphabet))
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

    # convert array of ints to one-hot vectors
    one_hots = torch.zeros(len(string), len(char_to_int))
    for i, j in enumerate(encoded_data):
        one_hots[i][j] = 1

    return one_hots


def cropBB(img, BB, size=None, fast=False):
    # augment coordinates
    # BB = augmentBB(BB)

    # crop
    if fast:
        BB = image_proc.order_points(BB)
        i,j = BB[0][1], BB[0][0]
        h,w = BB[2][1] - i, BB[2][0] - j

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


def simpleGenChar(N, img_dir, mat_dir, N_images=850000, matboundary=50000):
    # generate a sequence of N integers < 858750
    for i in random.sample(range(N_images), N):
        # determine matnum: i in gt_{matnum}.mat
        matnum = int(i/50000)
        j = i % matboundary

        matpath = os.path.join(mat_dir, f"gt_{matnum}.mat")
        # print(f"matpath = {matpath}") #

        f = h5py.File(matpath, 'r')
        string = "".join(image_proc.txtToInstance(f[f['txt'][j][0]]))
        k = random.randint(0, len(string)-1)
        # print(f"string = {string}\nk = {k}") #
        char = string[k]
        charBB = f[f['charBB'][j][0]][k]
        print(f"char = {char}") #

        imname = image_proc.u2ToStr(f[f['imnames'][j][0]])
        img_path = os.path.join(img_dir, imname)
        # print(f"img_path = {img_path}") #

        cropped = getCroppedImage(img_path, charBB)

        print("showing image...") #
        plt.figure() #
        plt.imshow(cropped, interpolation='nearest') #
        plt.show() #

        # yield cropped, char


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

# given matfile, entry number, character number:
# return image, character, image name
def getSample(f, entry_no, char_no, image_dir):
    imname = image_proc.u2ToStr(f[f['charBB'][entry_no][0]])
    image_path = os.path.join(image_dir, imname)

    charBB = f[f['charBB'][entry_no][0]][char_no]
    image = getCroppedImage(image_path, charBB)

    instances = image_proc.txtToInstance(f[f['txt'][entry_no][0]])
    string = ''.join(instances)
    char = string[char_no]

    return image, char, imname


if __name__ == '__main__':
    """
import numpy as np
import train_classifier
from importlib import reload

windows_path_prefix = "C:"
linux_path_prefix = "/mnt/A4B04DFEB04DD806"

path_prefix = linux_path_prefix
img_dir = path_prefix + '/Users/Aerjay/Downloads/SynthText/SynthText'
gt_path = path_prefix + '/Users/Aerjay/Downloads/SynthText/gt_v7.3.mat'
char_dir = path_prefix + '/Users/Aerjay/Downloads/SynthText/chars'

dataset = train_classifier.SynthCharDataset(gt_path, img_dir, (100,100))

r = dataset[0]
    """
    windows_path_prefix = "C:"
    linux_path_prefix = "/mnt/A4B04DFEB04DD806"

    path_prefix = linux_path_prefix
    img_dir = path_prefix + '/Users/Aerjay/Downloads/SynthText/SynthText'
    gt_path = path_prefix + '/Users/Aerjay/Downloads/SynthText/gt_v7.3.mat'
    char_dir = path_prefix + '/Users/Aerjay/Downloads/SynthText/chars'

    dataset = SynthCharDataset(gt_path, img_dir, (100,100))

    r = dataset[0]
    # classes = ['a', 'e', 'i', 'o', 'u']
    # simpleGenChar(10, img_dir, mat_dir, N_images=100)
    # genBalancedCharDataset(20, img_dir, mat_path, char_dir, classes=classes)