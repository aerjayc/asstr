import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import cv2

# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def perspectiveTransform(image, initial=None, final=None, size=None):
    if type(initial) == type(None):
        height, width = image.shape
        initial = np.array([[0, 0],
                            [width-1, 0],
                            [width-1, height-1],
                            [0, height-1]], dtype = "float32")
    if type(final) == type(None):
        maxWidth, maxHeight = getMaxSize(initial)
        final = np.array([[0, 0],
                          [maxWidth-1, 0],
                          [maxWidth-1, maxHeight-1],
                          [0, maxHeight-1]], dtype = "float32")
    if type(size) == type(None):
        w = max(final.T[0])
        h = max(final.T[1])
    else:
        h, w = size

    initial = order_points(initial)
    final = order_points(final)

    # print(f"initial = {initial}")
    # print(f"final = {final}")
    # print(f"Width = {w}\nHeight = {h}")

    M = cv2.getPerspectiveTransform(initial, final)
    warped = cv2.warpPerspective(image, M, (w, h))

    return warped

def getMaxSize(BBcoords):
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

def genDistortedGauss(BBcoords, img_size):
    """ BBcoords = 4x2 array
    """

    # make 2d isotropic Gaussian
    x_mean = 0
    y_mean = 0
    variance = 1
    height = np.sqrt(2*np.pi*variance)

    size = max(getMaxSize(BBcoords))
    bounds = 2.5

    x = np.linspace(-bounds, bounds, size)
    x, y = np.meshgrid(x,x)
    gauss = height * (1/np.sqrt(2*np.pi*variance)) * np.exp(-((x - x_mean)**2 + (y - y_mean)**2)/(2*variance))

    # plt.figure()
    # plt.imshow(gauss, interpolation='nearest')
    # plt.colorbar()

    distorted_gauss = perspectiveTransform(gauss, final=BBcoords, size=img_size)

    return distorted_gauss


if __name__ == '__main__':
    plt.figure()
    plt.imshow(cv2.imread('images/warped.jpg', 0), cmap='gray', interpolation='nearest')

    BBcoords = np.array([ (15,11),
                          (48,24),
                          (52,47),
                          (19,62) ])

    a = genDistortedGauss(BBcoords, img_size=(100,100))

    BBcoords = np.array([ (50,20),
                          (80,20),
                          (80,50),
                          (50,50) ])

    b = genDistortedGauss(BBcoords, img_size=(100,100))

    plt.figure()
    plt.imshow(a+b, interpolation='nearest')
    plt.colorbar()

    plt.show()
