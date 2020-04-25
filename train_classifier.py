import numpy as np
import cv2
import image_proc
import random
import h5py
import os.path
from pathlib import Path
import matplotlib.pyplot as plt


def genChar():
    # 1. randomly select character + GT from dataset by:
        # a. select matfile at random
        # b. select image file at random
        # c. select character at random
    # 2. use

    # get random integer i : 0 <= i <= 17
        # matfname = gt_{i}.mat
    # get random integer j : 0 <= j <= len(gt_i)
    # get random integer k : 0 <= k <= len(f[f['charBB'][i][0]])
    matpath = ''    # randomly selected
    f = h5py.File(matpath, 'r')
    charBB = f[f['charBB'][0][0]]
    pass
    # return cropped, char

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
    windows_path_prefix = "C:"
    linux_path_prefix = "/mnt/A4B04DFEB04DD806"

    path_prefix = linux_path_prefix
    img_dir = path_prefix + '/Users/Aerjay/Downloads/SynthText/SynthText'
    mat_path = path_prefix + '/Users/Aerjay/Downloads/SynthText/gt_v7.3.mat'
    char_dir = path_prefix + '/Users/Aerjay/Downloads/SynthText/chars'

    classes = ['a', 'e', 'i', 'o', 'u']
    # simpleGenChar(10, img_dir, mat_dir, N_images=100)
    genBalancedCharDataset(20, img_dir, mat_path, char_dir, classes=classes)