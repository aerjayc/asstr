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

    lines = {}
    skip_BBs = []
    line_num = 0
    for i, charBB in enumerate(charBBs):
        if i in skip_BBs:
            continue

        top = np.min(charBB, axis=0)[1]
        bottom = np.max(charBB, axis=0)[1]

        lines[line_num] = [i]

        for j, BB in enumerate(charBBs):
            if i == j:
                continue
            if j in skip_BBs:
                continue

            center_x, center_y = np.mean(BB, axis=0)

            if top <= center_y <= bottom:   # top has lower y-value
                # same line
                lines[line_num].append(j)
                skip_BBs.append(j)

        line_num += 1

    words = {}
    word_num = -1
    j = 0
    for i, line in enumerate(lines.values()):
        # get threshold for current line
        sum_width = 0
        for idx in line:
            charBB = charBBs[idx]
            width = ((charBB[1][0] - charBB[0][0]) + (charBB[2][0] - charBB[3][0])) / 2.0
            sum_width += width
        threshold = sum_width / len(line)

        # loop over all chars in current line
        prev_charBB = None
        for idx in line:
            charBB = charBBs[idx]
            if prev_charBB is None:
                word_num += 1
                words[word_num] = [j]
                j += 1
                prev_charBB = charBB
                continue

            right_prev = np.max(prev_charBB, axis=0)[0]
            left_curr = np.min(charBB, axis=0)[0]
            if (left_curr - right_prev) <= k*threshold:
                # same word
                words[word_num].append(j)
            else:
                # new word
                word_num += 1
                words[word_num] = [j]

            j += 1
            prev_charBB = charBB

    return lines, words


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