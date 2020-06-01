import numpy as np
import torch

import h5py

import image_proc


# General functions

def shuffle_in_unison(a, b):
    # source: https://stackoverflow.com/questions/4601373/
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def max_shape(arr_tensors):
    """Given a list of tensors, returns the maximum shape at each dimension
    """
    shape = np.zeros((len(arr_tensors), arr_tensors[0].dim()))
    for i, tensor in enumerate(arr_tensors):
        shape[i] = tensor.shape

    return list(np.max(shape, axis=0).astype("int32"))

def h5py_to_numpy(h5):
    """Convert h5py array of numpy arrays to a pure numpy array
    Args:
        h5 (h5py object): an h5py array of numpy arrays
    """
    sample = h5[0]

    numpy_arr = np.zeros([len(h5), *sample.shape], dtype=sample.dtype)
    for i, element in enumerate(h5):
        numpy_arr[i] = element

    return numpy_arr


# Functions on strings

def u2ToStr(u2, truncate_space=False):
    s = ""
    for c in u2:
        if truncate_space and c == 32:
            break
        else:
            s += chr(c)
    return s

def txtToInstance(txt):
    if isinstance(txt, h5py._hl.dataset.Dataset):
        txt_i = np.array(txt).T
        instances = []
        for instance in txt_i:
            instance = u2ToStr(instance)
            instances += instance.split()
    else:
        instances = "".join(txt).split()

    return instances

def getBreakpoints(txt):
    cumulative = -1
    breakpoints = []
    for instance in txtToInstance(txt):
        cumulative += len(instance)
        breakpoints += [cumulative]

    return breakpoints

def string_to_onehot(string, alphabet=None, char_to_int=None,
                     include_nonchar=True, to_onehot=True):
    """Converts a string to a onehot torch tensor
    """
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

    if to_onehot:
        # convert array of ints to one-hot vectors
        one_hots = torch.zeros(len(string), len(char_to_int))
        for i, j in enumerate(encoded_data):
            one_hots[i][j] = 1.0

        return one_hots
    else:
        return torch.Tensor(encoded_data)


# Functions on images / bounding boxes

def filter_BBs(BBs, gts, img_wh):
    """Filter BBs if they go out of the bounds of the image
        or if they have dimension <= 1
    Args:
        BBs (numpy.ndarray): a N x 4 x 2 array of bounding box coordinates
        gts (list): a list with the same length as BBs
        img_wh (2-tuple): a tuple (W, H) where W and H are the width and height
            (respectively) of the original image
    """
    img_w, img_h = img_wh
    skip_mask = np.ones(len(BBs), np.bool)
    correct_gts = []
    for i, BB in enumerate(BBs):
        tl, tr, br, bl = image_proc.get_containing_rect(BB)
        w, h = br - tl

        # if BB exceeds image bounds
        if (br[0] >= img_w) or (br[1] >= img_h):
            skip_mask[i] = 0
        # if BB dimensions <= 1
        elif (w <= 1) or (h <= 1):
            skip_mask[i] = 0
        # if no problems
        else:
            correct_gts.append(gts[i])

    return BBs[skip_mask], correct_gts

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

def BB_augment(charBBs, wordBBs, gts, img_wh, nonchars=False, shuffle=True,
               batch_size_limit=None, fraction_nonchar=0.1, expand_coeff=0.2,
               contract_coeff=0.2):
    if batch_size_limit is None:
        N = len(gts)
    elif batch_size_limit <= 0:
        N = len(gts)
        batch_size_limit = None
    else:
        N = min(len(gts), batch_size_limit) # batch size

    if nonchars:
        assert fraction_nonchar < 1,\
               f"fraction_nonchar = {fraction_nonchar}; must be < 1"

        # generate ``N_nonchars`` points not in wordBBs
        N_nonchars = int(fraction_nonchar*N)
        noncharBBs = genNonCharBBs(wordBBs, img_wh, N_nonchars, retries=10)

        shuffle_in_unison(charBBs, gts)
        charBBs, gts = charBBs[:N-N_nonchars], gts[:N-N_nonchars]

        # combine nonchars with chars
        if noncharBBs:
            charBBs = np.concatenate(charBBs, noncharBBs)
            gts += [None]*N_nonchars

        shuffle_in_unison(charBBs, gts)

    elif shuffle:
        shuffle_in_unison(charBBs, gts)

    # limit the inputs to ``batch_size_limit``
    # Note: list[:None] == list
    charBBs = charBBs[:batch_size_limit]
    gts = gts[:batch_size_limit]

    # perturb each coordinate in charBB, biased to increase the area
    for i, BB in enumerate(charBBs):
        if gts[i]:  # augmentation
            w_c, h_c = image_proc.get_width_height(BB)

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
            new_width, new_height = image_proc.get_width_height(BB)
            if (new_width > 1) and (new_height > 1):
                charBBs[i] = new_charBB

    # filter out faulty values
    charBBs, gts = filter_BBs(charBBs, gts, img_wh)

    return charBBs, gts

