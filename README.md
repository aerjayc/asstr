#### gt.mat Structure

gt.mat
    * f[name] | name = {'charBB', 'wordBB', 'immnames', 'txt'}
        * f[name][i][0] = hdf5 object reference; 0 < i < 8750
                        = index

    * f[ f['charBB' or 'wordBB'][i][0] ][j] = [charBBxy4_j]; 0 < j < N_chars or N_words
        * charBBxy4_j = [ (x0,y0),
                          (x1,y1),
                          (x2,y2),
                          (x3,y3) ]

    * f[ f['imnames'][i][0] ] = [ [fnamechar_0], [fnamechar_1], ... ] where char_i = ith char of fname
                              = imname
        * imname[j][0] = fnamechar_j

    *


    * charBB
        * shape: (8750, 1)
            *
    * wordBB
    * imnames
    * txt

    ``{'L': 171432, 'i': 1324921, 'n': 1601291, 'e': 3285706, 's': 1478527,
         ':': 599349, 'I': 245109, 'l': 823770, 'o': 1884072, 't': 2431977,
         'K': 26604, 'v': 212820, 'w': 534218, 'a': 2031519, 'd': 825882,
         'h': 1651242, '(': 61004, 'u': 724105, 'y': 499558, "'": 103873,
         'p': 398038, 'k': 199207, 'g': 364508, 'S': 205832, 'r': 1451002,
         'B': 82660, 'N': 117498, 'R': 147829, '-': 104635, 'T': 266968,
         'P': 78659, 'A': 225226, 'M': 147179, 'G': 56674, ',': 237698,
         '1': 187594, '8': 48077, 'c': 477372, 'F': 190562, 'm': 556960,
         '9': 108441, '5': 63787, '%': 3284, 'C': 91015, 'W': 61586, 'f': 462051,
         '>': 149052, 'b': 306481, 'D': 191958, 'Y': 32551, 'O': 81060, 'U': 36851,
         'H': 81017, '0': 105890, '.': 266182, '[': 4310, 'V': 24882, '2': 120258,
         'j': 51774, '3': 107707, '6': 56382, '?': 20489, ')': 31724, '"': 32459,
         '7': 41593, 'E': 75318, 'z': 28454, 'q': 10916, 'x': 47385, 'J': 42920,
         'X': 30465, '=': 1547, '_': 2332, '*': 6244, ']': 2995, 'Z': 3189, '4': 57217,
         ';': 5439, '&': 3075, '+': 2463, 'Q': 6569, '$': 11250, '@': 7708, '/': 10237,
         '|': 3972, '!': 9441, '<': 1989, '#': 3431, '`': 1166, '{': 549, '~': 469,
         '\\': 242, '}': 492, '^': 166}``


### Problems/notes

- dataset sample 196 produces warning `/home/eee198/Documents/ocr/asstr/image_proc.py:28: RuntimeWarning: invalid value encountered in float_scalars
  angle = np.arctan(y/x)`
    - 8/ballet_107_79.jpg

- dataset sample in [33600, 33700] produces error `error: OpenCV(3.4.2) /tmp/build/80754af9/opencv-suite_1535558553474/work/modules/imgproc/src/imgwarp.cpp:2902: error: (-215:Assertion failed) _src.total() > 0 in function 'warpPerspective'`

- dataset sample 1150 produces error: `error: OpenCV(3.4.2) /tmp/build/80754af9/opencv-suite_1535558553474/work/modules/imgproc/src/imgwarp.cpp:2902: error: (-215:Assertion failed) _src.total() > 0 in function 'warpPerspective'`
    - due to affinityBB

- dataset sample in [189, 199), [1479,1489) produces warning `/home/eee198/Documents/ocr/asstr/image_proc.py:24: RuntimeWarning: invalid value encountered in float_scalars angle = np.arctan(y/x)`
    - origin = (0,0), BBcoords = (0,0), occurs during regionBB

- minibatching will cause problems when the image shapes are not equal
    - padding
    - resizing (distortion)
- on the fly GT creation is a bottleneck
- no errors up to 10000


## Big Picture

1. ~~Char Map generation from Char BB~~ - *done*
    - Aerjay
2. Text Detector Training Code - *done* (needs cleaning)
    - ~~Timothy~~ Aerjay
3. Classifier Training Code - *ongoing*
    - ~~Aerjay~~ Timothy
4. Text Detector Training on Synthetic Data - *ongoing*
    - ~~Timothy~~ Aerjay
5. Classifier Training on Synthetic Data - *next*
    - ~~Aerjay~~ Timothy
6. Algorithm for Pseudo-GT Generation - *next*
    - ~~Timothy~~ Aerjay
7. ~~Char BB Generation from Char Maps~~ - *done*
    - ~~Aerjay~~ Timothy
8. Character-level Penalization Algorithm
    - Aerjay
9. Algorithm to deal with non-equal GT lengths
    - Timothy
10. Code for Text Detector Training
    - Aerjay
11. Classifier Training on Real-World Data
    - Timothy
12. Iterative Training of Detector and Classifier
    -


### To do

- prioritize char level loss
- the rest of the GT maps
    - make direction maps efficient
        - warp pre-made direction map to charBB instead of regenerating every time
- fix affinity maps bug
- sigmoid
    - note: can't be used directly on gt's with values outside [0,1] e.g. cos/sin maps
- train/~~val~~/test split (done)
- normalization (done)
- batch normalization
- make all functions use gpu
    - halving the output features should be done in gpu
- test overfitting (done)
- hard example mining
- ~~use float16~~
    - *use float32* instead, see https://stackoverflow.com/questions/24587433/
- check on `order_points` when pixels are too close (using pythagoras)


### Useful scripts:
- to count unique image bases: `ls -Rp | sed -n "s/\(.*\)_[0-9]*_[0-9]*\.jpg/\1/p" | sort | uniq | wc -l`
    - (110 images)
    - misleading, some distinct images share same base filename


### Teamviewer
- Partner ID: `735778156`
- Passcode: `62zp9v`