import numpy as np
import random
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path
import h5py
import cv2
import torch
import PIL
from torchvision import transforms
import scipy.ndimage
import torch.nn.functional as F
import numbers


# general, low-level image-processing functions
def wh2xy(x,y,w,h):
    return np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])

def genCharBB(heatmap, thresh=0.5, max_thresh=1, xywh=True):
    """
    Arguments
        heatmap (torch.Tensor or numpy.ndarray): must be of shape (H, W)
        thresh (number): threshold value (default = 0.5)
        max_thresh (number): value to be assigned to pixels that exceed thresh
                             (default = 1)
        xywh (Boolean): if True, the function will output a (n_BBs, 4)-shaped
                        numpy or torch tensor in the order x0,y0,w,h
                        else, the function will output a (n_BBs, 4, 2)-shaped
                        numpy or torch tensor
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.numpy()   # convert torch tensor to numpy array

    _, img_binary = cv2.threshold(heatmap, thresh, max_thresh,
                                  cv2.THRESH_BINARY)
    img_binary = img_binary.astype(np.uint8)

    contours, _ = cv2.findContours(img_binary, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)

    if xywh:
        BBs = np.zeros((len(contours), 4))
    else:
        BBs = np.zeros((len(contours), 4, 2))
    for i, contour in enumerate(contours):
        # get the bounding rect
        x0, y0, w, h = cv2.boundingRect(contour)

        if xywh:
            BBs[i] = (x0, y0, w, h)
        else:
            BBs[i] = wh2xy(x0, y0, w, h)

        # draw a green rectangle to visualize the bounding rect
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

    if isinstance(heatmap, torch.Tensor):
        BBs = torch.Tensor(BBs)

    return BBs

def getPixelAngle(origin, pt, units='radians'):
    x,y = (pt - origin).astype('float32')

    if x == 0:          # y-axis
        if y > 0:       # positive y-axis
            return np.pi/2
        elif y < 0:     # negative y-axis
            return -np.pi/2
        else:           # origin
            return 0
    else:               # not on x-axis
        angle = np.arctan(y/x)  # quadrant I
        if x < 0:       # left half
            angle += np.pi
        elif y < 0:     # quadrant IV
            angle += 2*np.pi

    if units != 'radians':
        angle *= 180/np.pi

    return angle

def getAffinityBB(charBB1, charBB2):
    edge1 = getAffinityEdge(charBB1)
    edge2 = getAffinityEdge(charBB2)
    return np.concatenate((edge1, edge2))

def getAffinityEdge(charBB):
    charBB = order_points(charBB)
    BB_center = getCentroid(charBB)
    tri1 = np.array([charBB[0], charBB[1], BB_center], dtype='float32')
    tri1_center = getCentroid(tri1)
    tri2 = np.array([charBB[2], charBB[3], BB_center], dtype='float32')
    tri2_center = getCentroid(tri2)

    return np.array([tri1_center, tri2_center], dtype='float32')

def getCentroid(vertices):
    return np.mean(vertices, axis=0)

def getMaxSize(BBcoords):
    # from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    (tl, tr, br, bl) = BBcoords

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    return maxWidth, maxHeight

def order_points(pts, warn=False):
    pts = pts.astype('float32')
    sorted_y = np.argsort(pts[:,1], axis=0).T

    top = pts[sorted_y[:2]]
    tl_i = np.argmin(top[:,0])  # find min x-value (i.e. leftmost) on top
    tl = top[tl_i]
    tr = top[1 - tl_i]

    bottom = pts[sorted_y[2:]]
    bl_i = np.argmin(bottom[:,0])  # find min x-value (i.e. leftmost) on bottom
    bl = bottom[bl_i]
    br = bottom[1 - bl_i]

    if warn:
        diff = pts[sorted_y[1]][1] - pts[sorted_y[2]][1]
        if np.abs(diff) < 1:
            #print(f"{diff}")
            print("Warning: BB coordinates difference < 1 px:")
            print(f"\ttl = {tl}\ttr = {tr}\n\tbl = {bl}\tbr = {br}\n")
    return np.array([tl,tr,br,bl], dtype='float32')


# high-level image-processing functions
def perspectiveTransform(image, initial=None, final=None, size=None):
    if initial is None:
        height, width = image.shape
        initial = np.array([[0, 0],
                            [width-1, 0],
                            [width-1, height-1],
                            [0, height-1]], dtype = "float32")
    if final is None:
        maxWidth, maxHeight = getMaxSize(initial)
        final = np.array([[0, 0],
                          [maxWidth-1, 0],
                          [maxWidth-1, maxHeight-1],
                          [0, maxHeight-1]], dtype = "float32")
    if size is None:
        w = max(final.T[0])
        h = max(final.T[1])
    else:
        h, w = size

    initial = order_points(initial)
    final = order_points(final)

    assert (w > 0) and (h > 0), f"perspectiveTransform error: w = {w}, h = {h}"

    M = cv2.getPerspectiveTransform(initial, final)
    warped = cv2.warpPerspective(image, M, (w, h))

    return warped.astype('float32')

def genDistortedGauss(BBcoords, img_size, template=None):
    """ Using a pre-made template increases efficiency.
        From `~210 us` to `~90 us` (execution duration, timed using `%timeit`)
    """
    if template is None:
        size = max(getMaxSize(BBcoords))
        if not size:
            return None

        x_mean = 0
        y_mean = 0
        variance = 1
        height = np.sqrt(2*np.pi*variance)

        bounds = 2.5

        x = np.linspace(-bounds, bounds, size, dtype='float32')
        x, y = np.meshgrid(x,x)#.astype('float32')
        gauss = height * (1/np.sqrt(2*np.pi*variance)) *\
                np.exp(-((x - x_mean)**2 + (y - y_mean)**2)/(2*variance))
    else:
        gauss = template

    distorted_gauss = perspectiveTransform(gauss, final=BBcoords, size=img_size)

    return distorted_gauss.astype('float32')

def genGaussianTemplate(size=None):
    if size is None:
        size = (32,32)
    h,w = size

    BB = np.array([[0,0],[w,0],[w,h],[0,h]])

    return genDistortedGauss(BB, size)

def genPseudoGT(charBB_i, txt, image_shape, generate_affinity=True, template=None):
    assert charBB_i.shape[-2:] == (4,2)
    if charBB_i.ndim == 2:  # if only one char
        charBB_i = np.array([charBB_i], dtype='float32')

    if generate_affinity:
        import dataset_utils
        breakpoints = dataset_utils.getBreakpoints(txt)

    pseudoGT_blank = np.zeros(image_shape, dtype='float32')
    pseudoGT_region = pseudoGT_blank.copy()
    pseudoGT_affinity = pseudoGT_blank.copy()
    charBB_prev = None
    for j, charBB in enumerate(charBB_i):
        region_mask = genDistortedGauss(charBB, img_size=image_shape,
                                        template=template)
        if not (region_mask is None):
            pseudoGT_region += region_mask

        if not generate_affinity:
            continue
        # if prev char is not a breakpoint = if curr char same instance as prev
        if ((j-1) not in breakpoints) and (j > 0) and generate_affinity:
            if region_mask is None:         # if previous char (assuming same
                continue                    # instance) has zero size

            affinityBB = order_points(getAffinityBB(charBB_prev, charBB))
            affinity_mask = genDistortedGauss(affinityBB, img_size=image_shape,
                                              template=template)
            if not (affinity_mask is None):
                pseudoGT_affinity += affinity_mask
        charBB_prev = charBB

    if not generate_affinity:
        pseudoGT_affinity = None

    return pseudoGT_region, pseudoGT_affinity

def genWordGT(wordBB_i, image_shape, template=None):
    assert wordBB_i.shape[-2:] == (4,2)
    if wordBB_i.ndim == 2:  # if only one word
        wordBB_i = np.array([wordBB_i], dtype='float32')

    mask = np.zeros(image_shape, dtype='float32')

    for j, wordBB in enumerate(wordBB_i):
        mask += genDistortedGauss(wordBB, img_size=image_shape,
                                  template=template)

    return mask.astype('float32')


# Efficient Direction GT
def genDirectionMapTemplate(temp_shape=(32,32)):
    temp_h, temp_w = temp_shape
    tempBB = np.array([ [0     , 0     ],
                        [temp_w, 0     ],
                        [temp_w, temp_h],
                        [0     , temp_h] ], dtype="uint8")

    cos_temp, sin_temp = genDirectionMap(tempBB, (temp_w, temp_h))
    return cos_temp, sin_temp


def genDirectionGT(BBs, img_size, template=None, normalize=False):
    """~7 times faster (excluding the template generation time)
        ``` In [85]: %timeit image_proc.genDirectionGT_efficient(BB, (10,10), template=t)
                207 µs ± 510 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)

            In [86]: %timeit image_proc.genDirectionGT(BB, (10,10))
                1.59 ms ± 3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
        ```
    """
    if template is None:
        template = genDirectionMapTemplate()

    if BBs.ndim == 2:    # if only one BB
        BBs = np.array([BBs], dtype="float32")

    cos_mask = np.zeros(img_size, dtype="float32")
    sin_mask = cos_mask.copy()
    for BB in BBs:
        cos_mask += perspectiveTransform(template[0], final=BB, size=img_size)
        sin_mask += perspectiveTransform(template[1], final=BB, size=img_size)

    np.clip(cos_mask, -1, 1, out=cos_mask)
    np.clip(sin_mask, -1, 1, out=sin_mask)
    if normalize:
        cos_mask = (cos_mask + 1) / 2
        sin_mask = (sin_mask + 1) / 2

    return cos_mask, sin_mask


# # Direction GT
# def genDirectionGT(charBB_i, img_size):
#     assert charBB_i.shape[-2:] == (4,2)
#     if charBB_i.ndim == 2:  # if only one char
#         charBB_i = np.array([charBB_i], dtype='float32')

#     cosf = np.zeros(img_size, dtype='float32')
#     sinf = np.zeros(img_size, dtype='float32')
#     for charBB in charBB_i:
#         cos_angle, sin_angle = genDirectionMap(charBB, img_size)
#         cosf += cos_angle
#         sinf += sin_angle

#     return cosf, sinf

def genDirectionMap(charBB, img_size):
    centroid = getCentroid(charBB)
    x_min, y_min = np.min(charBB, axis=0)
    x_max, y_max = np.max(charBB, axis=0)

    # https://stackoverflow.com/questions/21339448
    x,y = np.meshgrid(np.arange(x_min,x_max), np.arange(y_min,y_max))
    x,y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).astype('int32').T

    sin_field = np.zeros(img_size, dtype='float32')
    cos_field = np.zeros(img_size, dtype='float32')
    # angle_field = np.zeros(img_size)

    p = Path(charBB)
    for pt in points:
        if p.contains_point(pt):
            angle = getPixelAngle(centroid, pt)
            # angle_field[pt[1], pt[0]] = angle
            cos_field[pt[1], pt[0]] = np.cos(angle).astype("float32")
            sin_field[pt[1], pt[0]] = np.sin(angle).astype("float32")

    return cos_field, sin_field

def zero_pad(tensors, shape=None, cuda=True):
    """ Input:
            `tensors`:  a list of (C,h,w)-shaped tensors
                        (variable h,w)
        Output is shaped (N,C,H,W) where h,w = max(h), max(w)
    """
    if shape is None:
        max_shape = np.max([tensor.shape for tensor in tensors], axis=0)\
                        .astype("int32").tolist()
        if len(max_shape) == 4: # if batched already (e.g. NCHW)
            N = sum([len(tensor) for tensor in tensors])
            batch_shape = [N,] + max_shape[-3:]
        else:
            N = len(tensors)
            batch_shape = [N,] + max_shape

    template = torch.zeros(batch_shape)
    i = 0
    for tensor in tensors:
        s = tensor.shape
        if len(s) == 4:
            i_end = i + s[0]
        else:
            i_end = i + 1
        template[i:i_end, :s[-3], :s[-2], :s[-1]] = tensor

        i = i_end

    if cuda: template = template.cuda()

    return template


def centroid2minmax(centroids, shapes, yx_max):
    assert shapes.shape[-1:] == (2,), f"shapes.shape = {shapes.shape}"
    if shapes.shape == (2,):
        shapes = np.array([shapes], dtype="int32")

    centroids = centroids.astype("int32")
    shapes = shapes.astype("int32")

    BB_min = (centroids - 0.5*shapes).astype("int32")
    BB_max = BB_min + shapes


    # remove under while preserving shape
    if np.any(BB_min < 0):
        BB_min_copy = BB_min.copy()

        BB_min_copy[BB_min > 0] = 0
        BB_max -= BB_min_copy

        np.clip(BB_min, 0, None, BB_min)

    if np.any(BB_max > yx_max):
        BB_max_copy = BB_max.copy()
        BB_max_copy -= yx_max
        BB_max_copy[BB_max_copy < 0] = 0
        BB_min -= BB_max_copy

        np.clip(BB_max, None, yx_max, BB_max)

    return np.array([BB_min, BB_max], dtype="int32")


def hard_example_mining(img, gt, wordBBs, N_examples=4, constant_hw=True):
    """ Inputs:
            - `img`: PIL Image
            - `gt`: must have the same width and height as `img`
                    i.e. a numpy array shaped (..., H, W)
            - `wordBBs`: numpy array shaped `(N_words, H, W)`
                         Note: `wordBBs` *has x-y order!*
            - `N_examples`: A positive integer (default: `4`)
                            The number of crops to be done (batch_size)
            - `constant_hw`:
        randomly generate BB region to crop
            parameters:
            - size/shape,
            - number of characters inside,
            - number of crops to do (minimize intersection)
            - random scaling
    """

    # generate random BBs based on word locations
    # fliplr makes the order y-x instead of x-y
    if wordBBs.shape == (4,2):  # if single BB
        wordBBs = wordBBs[None,...]
    word_centroids = np.mean(wordBBs, axis=1)
    # pick `N_examples` points from `centroids`
    N_examples = min(len(wordBBs), N_examples)  # make sure that N_examples <= # BBs
    indices = random.sample(range(len(wordBBs)), N_examples)
    BB_centroids = np.fliplr(word_centroids[indices])

    # pick semi-random width & height (only once)
    img_height, img_width = img.height, img.width
    scaling_factors = np.array([0.5, 0.3, 0.25, 0.2], dtype="float32")
    cropped_heights = (scaling_factors*img_height).astype("int32")
    cropped_widths = (scaling_factors*img_width).astype("int32")

    if constant_hw:
        N_examples = 1

    # generate `2*N_examples` ints (from `dimensions`) to get `N_examples` h,w
    cropped_height = np.random.choice(cropped_heights, size=N_examples)
    cropped_width = np.random.choice(cropped_widths, size=N_examples)
    shapes = np.array([[cropped_height, cropped_width]], dtype="int32")\
                .reshape(-1,2)

    # combine img and gts for efficiency
    img = np.array(img).transpose(2,0,1)
    imgt = np.concatenate((img, gt))
    # crop
    cropped_imgts = constantShapeCrop(imgt, BB_centroids, shapes)
    # separate cropped_images from cropped_gts
    C = len(img)
    cropped_images = cropped_imgts[:, :C, :, :] # NCHW
    cropped_gts = cropped_imgts[:, C:, :, :]    # NCHW

    return cropped_images, cropped_gts


def constantShapeCrop(img, centroids, shapes):
    """ Inputs:
            - `img`: numpy tensor with shape (C, H, W)
            - `yx_mins`: yx coordinates of top left corner of BB
            - `shape`: (height, width) of the BB

        Outputs:
            - `(N, H, W)`-shaped numpy tensor where
                - `N`: number of BBs cropped (given by `len(yx_mins)`)
                - `H, W`: shape of the resulting BB (given by `shape`)
    """
    yx_mins, yx_maxs = centroid2minmax(centroids, shapes, img.shape[-2:])

    batch = None
    for (y_min, x_min), (y_max, x_max) in zip(yx_mins, yx_maxs):
        cropped = img[None, ..., y_min:y_max, x_min:x_max]

        if batch is None:
            batch = cropped
            continue

        batch = np.concatenate((batch, cropped))

    return batch

def augment(img, gt, size=None, scale=(0.08, 1.0), ratio=(3./4, 4./3),
            degrees=[-180,180], gt_hw_axes=[-2,-1], halve_gt=False):
    """Performs data augmentation (transforms) on the img and gt.
    Args:
        img (PIL Image):
        gt (numpy array): CHW
        size (tuple or int or None):
    Returns:
        t_img (torch Tensor): CHW
        t_gt (torch Tensor):
    """
    if isinstance(size, numbers.Number):
        size = (size, size)
    # random parameter generation
    i,j,h,w = transforms.RandomResizedCrop.get_params(img, scale, ratio)
    angle = transforms.RandomRotation.get_params(degrees)

    # rotate
    img = scipy.ndimage.rotate(img, angle, reshape=False)
    gt  = scipy.ndimage.rotate(gt, angle, axes=gt_hw_axes, reshape=False)

    # convert to torch Tensor and crop
    img = crop(img, i,j,h,w)    # HWC
    gt  = crop(gt, i,j,h,w, axes=gt_hw_axes)   # CHW

    # resize
    if size:
        img = cv2.resize(img, dsize=size)
        # resize(gt->HWC)->CHW
        gt  = cv2.resize(gt.transpose(1,2,0), dsize=size)

        if gt.ndim == 2:    # if only 1 map, cv2.resize returns 2 dim
            gt = gt[None, ...]
        else:
            gt = gt.transpose(2,0,1)

    # convert to torch tensor
    img = torch.Tensor(img).permute(2,0,1)  # CHW
    gt = torch.Tensor(gt)                   # CHW

    # halve gt
    if halve_gt:
        gt = F.interpolate(gt[None,...], scale_factor=0.5)[0]

    return img, gt


def crop(tensor, i,j,h,w, axes=[0,1]):
    """ crops tensor
    Args:
        tensor (torch Tensor, PIL Image, or numpy array): tensor to be cropped
            - if torch.Tensor and axes=None: assumes shape of (...,H,W)
            - if PIL.Image.Image: ``axes`` is ignored
            - if numpy.ndarray and axes=None: assumes shape of (H,W,...)
            - if axes is not None: assume ``axes`` are the indices of H,W
                respectively, except if tensor is a PIL Image
        i,j,h,w (numbers): x_topleft, y_topleft, height, width respectively
            - converted to integers (truncate fractional part) before usage
        axes (list-type of length 2): indices of the height and width channels,
            respectively
    """
    if isinstance(tensor, PIL.Image.Image):
        tensor = np.array(tensor)   # (H,W,C)
    elif isinstance(tensor, torch.Tensor):
        if axes is None:
            axes = [-2,-1]  # assume shape of (...,H,W)
        tensor = np.array(tensor)
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        print("``crop`` only accepts inputs of type: ``torch.Tensor``, " +
              "``PIL.Image.Image``, or ``numpy.ndarray``")
        return None

    i,j,h,w = int(i), int(j), int(h), int(w)

    # convert all negative indices to nonnegative equivalent
    axes = [axis % tensor.ndim for axis in axes]

    # get axis order
    axes_set = set(range(tensor.ndim))
    channels = list(axes_set - set(axes))
    axis_order = axes + channels

    # order axes to HW...
    tensor = tensor.transpose(axis_order)
    # crop
    tensor = tensor[i:i+h, j:j+w, ...]
    # reorder axes back to previous configuration
    tensor = tensor.transpose(np.argsort(axis_order))
    return tensor

def cropBB(img, BB, size=None, fast=False):
    # augment coordinates
    # BB = augmentBB(BB)

    # crop
    if fast:
        BB = order_points(BB)
        j, i = np.min(BB, axis=0)
        # use ceil in w,h to prevent w=0 or h=0
        w, h = np.ceil(np.max(BB, axis=0) - np.min(BB, axis=0))

        cropped = crop(img, i,j,h,w)
    else:
        if isinstance(img, PIL.Image.Image):
            img = np.array(img, dtype='float32')

        cropped = perspectiveTransform(img, initial=BB, size=size)

    return cropped

def get_containing_rect(points):
    """Get the BB coordinates of the smallest rectangle containing ``points``

    Finds the smallest horizontal rectangle containing
    a set of points given by ``points``.

    Args:
        points (numpy.ndarray): a N x 2 numpy array
    """
    points = order_points(points)

    tl = np.min(points, axis=0)
    br = np.max(points, axis=0)
    tr = np.array([br[0], tl[1]])
    bl = np.array([tl[0], br[1]])

    return np.array([tl, tr, br, bl])

def get_width_height(points):
    """Get the dimensions of the smallest rectangle containing ``points``.

    Finds the smallest horizontal rectangle that contains
    the coordinates given by ``points``.

    Args:
        points (numpy.ndarray): a N x 2 numpy array
    """
    top_left = np.min(points, axis=0)
    bottom_right = np.max(points, axis=0)

    width, height = bottom_right - top_left

    return width, height


if __name__ == '__main__':
    pass

