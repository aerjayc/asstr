import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path
import cv2

# matfile-specific functions
def u2ToStr(u2, truncate_space=False):
    s = ""
    for c in u2:
        if truncate_space and c == 32:
            break
        else:
            s += chr(c)
    return s

# general, low-level image-processing functions
def getPixelAngle(origin, pt, units='radians'):
    x,y = pt - origin

    # if (x == 0) and (y == 0):
    #     print("warning: x=y=0")
    #     print(f"origin = {origin}\tpt = {pt}")
    #print(f"x,y = {x}, {y}")
    if x == 0:
        angle = np.pi/2
    angle = np.arctan(y/x)
    if x < 0: # quadrant II and III
        angle += np.pi
    elif y < 0:   # quadrant IV
        angle += 2*np.pi

    if units != 'radians':
        angle *= 180/np.pi

    return angle

def getPixelAngles(origin, pts, units='radians'):
    angles = np.zeros(pts.shape[:-1], dtype='float32')
    angles = angles.reshape((-1,1))

    # print("getting pixel angle")
    for i, pt in enumerate(pts.reshape((-1,2))):
        # if np
        angles[i] = getPixelAngle(origin, pt, units=units)
    
    angles = angles.reshape(pts.shape[:-1])

    return angles

def getAffinityBB(charBB1, charBB2):
    edge1 = getAffinityEdge(charBB1)
    edge2 = getAffinityEdge(charBB2)
    return np.concatenate((edge1, edge2))

def getAffinityEdge(charBB):
    charBB = order_points(charBB)
    BB_center = getCentroid(charBB)
    tri1 = np.array([charBB[0], charBB[1], BB_center])
    tri1_center = getCentroid(tri1)
    tri2 = np.array([charBB[2], charBB[3], BB_center])
    tri2_center = getCentroid(tri2)

    return np.array([tri1_center, tri2_center])

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

def order_points(pts):
    centroid = getCentroid(pts)
    angles = getPixelAngles(centroid, pts)
    for i, angle in enumerate(angles):  # take 90 degrees as min to get
        if 0 <= angle <= np.pi/2:       # top left vertex as initial pt
            angles[i] += 2*np.pi

    # sort by pts by angle (descending)
    ordered_pts = pts[len(pts)-1 - np.argsort(angles)]
    # put last pt (least angle i.e. pt nearest to y-axis at QII) to start
    ordered_pts = ordered_pts[np.arange(len(pts))-1].astype('float32')

    return ordered_pts

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

    M = cv2.getPerspectiveTransform(initial, final)
    warped = cv2.warpPerspective(image, M, (w, h))

    return warped

def genDistortedGauss(BBcoords, img_size):
    size = max(getMaxSize(BBcoords))
    if not size:
        return None

    x_mean = 0
    y_mean = 0
    variance = 1
    height = np.sqrt(2*np.pi*variance)

    bounds = 2.5

    x = np.linspace(-bounds, bounds, size)
    x, y = np.meshgrid(x,x)
    gauss = height * (1/np.sqrt(2*np.pi*variance)) * np.exp(-((x - x_mean)**2 + (y - y_mean)**2)/(2*variance))

    # plt.figure()
    # plt.imshow(gauss, interpolation='nearest')
    # plt.colorbar()

    distorted_gauss = perspectiveTransform(gauss, final=BBcoords, size=img_size)

    return distorted_gauss

def getBreakpoints(txt):
    cumulative = -1
    breakpoints = []
    for instance in txtToInstance(txt):
        cumulative += len(instance)
        breakpoints += [cumulative]

    return breakpoints

def txtToInstance(txt):
    txt_i = np.array(txt).T
    instances = []
    for instance in txt_i:
        instance = u2ToStr(instance)
        instances += instance.split()
    
    return instances

def genPseudoGT(charBB_i, txt, image_shape, generate_affinity=True):
    breakpoints = getBreakpoints(txt)
    # instances = txtToInstance(txt)
    # entire_string = ''.join(instances)

    pseudoGT_blank = np.zeros(image_shape, dtype='float')
    pseudoGT_region = pseudoGT_blank.copy()
    pseudoGT_affinity = pseudoGT_blank.copy()
    charBB_prev = None
    for j, charBB in enumerate(charBB_i):
        region_mask = genDistortedGauss(charBB, img_size=image_shape)
        if not (region_mask is None):
            pseudoGT_region += region_mask
        
        # if prev char is not a breakpoint = if curr char same instance as prev
        if ((j-1) not in breakpoints) and (j > 0) and generate_affinity:
            if region_mask is None:    # if previous char (assuming same instance) has zero size
                continue

            affinityBB = order_points(getAffinityBB(charBB_prev, charBB))
            affinity_mask = genDistortedGauss(affinityBB, img_size=image_shape)
            if not (affinity_mask is None):
                pseudoGT_affinity += affinity_mask
        charBB_prev = charBB

    if not generate_affinity:
        pseudoGT_affinity = None
    
    return pseudoGT_region, pseudoGT_affinity

# Direction GT
def genDirectionGT(charBB_i, img_size):
    cosf = np.zeros(img_size)
    sinf = np.zeros(img_size)
    for charBB in charBB_i:
        cos_angle, sin_angle = genDirectionMap(charBB, img_size)
        cosf += cos_angle
        sinf += sin_angle

    return cosf, sinf

def genDirectionMap(charBB, img_size):
    centroid = getCentroid(charBB)
    x_min, y_min = np.min(charBB, axis=0)
    x_max, y_max = np.max(charBB, axis=0)

    # https://stackoverflow.com/questions/21339448/how-to-get-list-of-points-inside-a-polygon-in-python
    x,y = np.meshgrid(np.arange(x_min,x_max), np.arange(y_min,y_max))
    x,y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).astype('int32').T

    sin_field = np.zeros(img_size)
    cos_field = np.zeros(img_size)
    # angle_field = np.zeros(img_size)

    p = Path(charBB)
    for pt in points:
        if p.contains_point(pt):
            angle = getPixelAngle(centroid, pt)
            # angle_field[pt[1], pt[0]] = angle
            cos_field[pt[1], pt[0]] = np.cos(angle)
            sin_field[pt[1], pt[0]] = np.sin(angle)

    return cos_field, sin_field


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
