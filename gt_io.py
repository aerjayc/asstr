import numpy as np
import h5py


def u2ToStr(u2, truncate_space=False):
    if truncate_space:
        s = ""
        for c in u2:
            if c == 32:
                break
            else:
                s += chr(c)
        return s
    return "".join([chr(c[0]) for c in u2])

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
        # filename
        imname_i = f[f['imnames'][i][0]]
        imname = u2ToStr(imname_i)

        # charBB's in imname_i
        charBB_i = f[f['charBB'][i][0]]

        # wordBB's in imname_i
        wordBB_i = f[f['wordBB'][i][0]]


        print(f"filename: {imname}")
        print(f"word lengths: {len}")
        print("Text:")
        # txt's in imname_i
        txt_i = np.array(f[f['txt'][i][0]]).T
        for instance in txt_i:
            str_instance = u2ToStr(instance, truncate_space=True)
            print(repr(f"\t{str_instance}"))

        break


if __name__ == '__main__':
    # getCharBBxy('gt_17.mat')
    pass