import numpy as np
import cv2
import os.path
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import image_proc


if __name__ == '__main__':
    skip_existing = False
    matpath = 'C:/Users/Aerjay/Downloads/SynthText/parts/gt_0.mat'
    imgdir = 'C:/Users/Aerjay/Downloads/SynthText/SynthText/'
    pseudoGT_dir = 'C:/Users/Aerjay/Downloads/SynthText/pseudoGT/'

    f = h5py.File(matpath, 'r')

    # https://stackoverflow.com/questions/28541847/
    N = len(f['imnames'])
    for i in range(0, N):
        # filename
        imname_i = f[f['imnames'][i][0]]
        imname = image_proc.u2ToStr(imname_i)
        print(f"file: {imname}")
        
        pseudoGT_prefix, pseudoGT_ext = os.path.splitext(imname)
        pseudoGT_name = pseudoGT_prefix + '_region.png' #+ pseudoGT_ext
        pseudoGT_path = os.path.join(pseudoGT_dir, pseudoGT_name)
        if skip_existing and os.path.isfile(pseudoGT_path):
            print("\tAlready exists. Skipping...")
            continue

        # image shape
        imgpath = os.path.join(imgdir, imname)
        image_shape = cv2.imread(imgpath).shape[0:2]

        # charBB's in imname_i
        charBB_i = f[f['charBB'][i][0]]
        # text in imname_i
        txt_i = f[f['txt'][i][0]]
        pseudoGT_region, pseudoGT_affinity = image_proc.genPseudoGT(charBB_i, txt_i, image_shape)

        Path(os.path.split(pseudoGT_path)[0]).mkdir(parents=True, exist_ok=True)
        print(f"\tsaving to {pseudoGT_path}")
        cv2.imwrite(pseudoGT_path, pseudoGT_region*255)
        # print("showing image...")
        # plt.figure()
        # plt.imshow(pseudoGT_region, interpolation='nearest')
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(pseudoGT_affinity, interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        # break