import numpy as np
import pandas as pd
import PIL.Image

def character_grouper(charBBs, k=0.7):
    """BBs: numpy array with shape N,4,2
        Algorithm:
            # to get lines
            for charBB in charBBs:
                line = BB in charBBs | bottom(charBB) <= center_y(BB) <= top(charBB)

            # to get words in a line
            THRESH = average char width in line
            k = 0.5
            for char in line:
                if left_char - right_prevBB <= k*THRESH:
                    same word
    """

    skip_BBs = []
    line_num = 0
    lines = []
    for i, charBB in enumerate(charBBs):
        if i in skip_BBs:
            continue

        top = np.min(charBB, axis=0)[1]
        bottom = np.max(charBB, axis=0)[1]

        lines.append([charBB])

        for j, BB in enumerate(charBBs):
            if i == j:
                continue
            if j in skip_BBs:
                continue

            center_x, center_y = np.mean(BB, axis=0)

            if top <= center_y <= bottom:   # top has lower y-value
                # same line
                skip_BBs.append(j)
                lines[line_num].append(BB)

        lines[line_num] = np.array(lines[line_num])
        line_num += 1

    # sort lines
    # line_ys = [np.mean(line.reshape(-1,2), axis=0)[1] for line in lines]
    # lines = lines[np.argsort(line_ys)]

    # sort chars in each line    # line: (n_chars, 4, 2)
    for i, line in enumerate(lines):
        sort_x = line[:,0,0].argsort()
        lines[i] = line[sort_x]

    words = []
    words_idx = []
    word_num = -1
    char_num = 0
    for i, line in enumerate(lines):
        # get threshold for current line
        sum_width = 0
        for charBB in line:     # get average width in line
            width = ((charBB[1][0] - charBB[0][0]) +
                     (charBB[2][0] - charBB[3][0])) / 2.0
            sum_width += width
        threshold = sum_width / len(line)

        # loop over all chars in current line
        prev_charBB = None
        for charBB in line:
            if prev_charBB is None:     # first iteration
                word_num += 1
                words.append([charBB])
                words_idx.append([char_num])
                char_num += 1
                prev_charBB = charBB
                continue

            right_prev = np.max(prev_charBB, axis=0)[0]
            left_curr = np.min(charBB, axis=0)[0]
            if (left_curr - right_prev) <= k*threshold:
                # same word
                words[word_num].append(charBB)
                words_idx[word_num].append(char_num)
            else:
                # new word
                word_num += 1
                words.append([charBB])
                words_idx.append([char_num])

            char_num += 1
            prev_charBB = charBB

    for i in range(len(words)):
        words[i] = np.array(words[i])

    return lines, words, words_idx

def words_to_wordBBs(words):
    wordBBs = np.zeros((len(words), 4, 2))
    for i, word in enumerate(words):
        coords = word.reshape(-1,2)
        tl = np.min(coords, axis=0)
        br = np.max(coords, axis=0)
        tr = np.array([br[0], tl[1]])
        bl = np.array([tl[0], br[1]])

        wordBBs[i] = np.array([tl, tr, br, bl])

    return wordBBs.astype('int')

def main(i=0):
    headers = ['R', 'G' ,'B' ,  # RGB values
                   'x0', 'y0',      # center
                   'x1', 'y1',      # top left
                   'x2', 'y2',      # bottom right
                   'character']
    gt_df = pd.read_csv(gt_path, names=headers, comment='#',
                        delim_whitespace=True, doublequote=False)

    charBBs = gt_df[['x1', 'y1',
                     'x2', 'y1',
                     'x2', 'y2',
                     'x1', 'y2']].to_numpy().reshape(-1,4,2)

    img = PIL.Image.open(img_path)

    return img, charBBs

if __name__ == '__main__':
    main()
