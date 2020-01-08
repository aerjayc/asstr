import numpy as np
import cv2
import os.path
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import image_proc


def u2ToStr(u2, truncate_space=False):
    # if truncate_space:
    #     s = ""
    #     for c in u2:
    #         if c == 32:
    #             break
    #         else:
    #             s += chr(c)
    #     return s

    s = ""
    for c in u2:
        if truncate_space and c == 32:
            break
        else:
            s += chr(c)
    return s
    #return "".join([chr(c[0]) for c in u2])

def getCharBBxy(matpath):
    """ 3. charBB  : character-level bounding-boxes,
                    each represented by a tensor of size 2x4xNCHARS_i
                    - the first dimension is 2 for x and y respectively,
                    - the second dimension corresponds to the 4 points
                        (clockwise, starting from top-left), and
                    -  the third dimension of size NCHARS_i, corresponds to
                        the number of characters in the i_th image.
    """

    f = h5py.File(matpath, 'r')

    # https://stackoverflow.com/questions/28541847/
    N = len(f['imnames'])
    for i in range(0, N):
        i = 10000
        # filename
        imname_i = f[f['imnames'][i][0]]
        imname = u2ToStr(imname_i)

        # charBB's in imname_i
        charBB_i = f[f['charBB'][i][0]]

        # wordBB's in imname_i
        wordBB_i = f[f['wordBB'][i][0]]


        print(f"filename: {imname}")
        #print(f"word lengths: {len(charBB_i)}")
        print("Text:")
        # txt's in imname_i
        txt_i = np.array(f[f['txt'][i][0]]).T
        for instance in txt_i:
            str_instance = u2ToStr(instance, truncate_space=False)
            print("\t" + repr(str_instance))

        break


if __name__ == '__main__':
    matpath = 'C:/Users/Aerjay/Downloads/SynthText/parts/gt_0.mat'
    imgdir = 'C:/Users/Aerjay/Downloads/SynthText/SynthText/'
    pseudoGT_dir = 'C:/Users/Aerjay/Downloads/SynthText/pseudoGT/'

    f = h5py.File(matpath, 'r')

    # https://stackoverflow.com/questions/28541847/
    N = len(f['imnames'])
    for i in range(0, N):
        # filename
        imname_i = f[f['imnames'][i][0]]
        imname = u2ToStr(imname_i)
        print(f"file: {imname}")
        
        pseudoGT_prefix, pseudoGT_ext = os.path.splitext(imname)
        pseudoGT_name = pseudoGT_prefix + '_region' + pseudoGT_ext
        pseudoGT_path = os.path.join(pseudoGT_dir, pseudoGT_name)
        if os.path.isfile(pseudoGT_path):
            print("\tAlready exists. Skipping...")
            continue

        # image shape
        imgpath = os.path.join(imgdir, imname)
        image_shape = cv2.imread(imgpath).shape[0:2]
        # print(f"shape: (h,w) = {image_shape}")

        # charBB's in imname_i
        charBB_i = f[f['charBB'][i][0]]
        pseudoGT_blank = np.zeros(image_shape)
        pseudoGT_region = pseudoGT_blank.copy()
        for charBB in charBB_i:
            pseudoGT_region += image_proc.genDistortedGauss(charBB, img_size=image_shape)

        Path(os.path.split(pseudoGT_path)[0]).mkdir(parents=True, exist_ok=True)
        print(f"\tsaving to {pseudoGT_path}")
        cv2.imwrite(pseudoGT_path, pseudoGT_region*255)
        # print("showing image...")
        # plt.figure()
        # plt.imshow(pseudoGT_region, interpolation='nearest')
        # plt.colorbar()
        # plt.show()