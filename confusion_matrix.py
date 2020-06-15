import numpy as np
import sklearn.metrics
from itertools import combinations
from scipy.special import comb
import re
import datasets

def str_to_alphabet(word, alphabet, null=None):
    new_w = []
    for c in word:
        if c is None:
            c = null
        elif c not in alphabet:
            if c.upper() in alphabet:
                c = c.upper()
            else:
                c = null
        new_w.append(c)

    return new_w

def get_best_fit(w1, w2, alphabet, null=' '):
    w1 = str_to_alphabet(w1, alphabet, null=null)
    w2 = str_to_alphabet(w2, alphabet, null=null)

    n = max(len(w1), len(w2))
    k = min(len(w1), len(w2))

    max_score = 0
    fit_w1 = list(w1[:k])
    fit_w2 = list(w2[:k])

    if comb(n, k) > 1e5:    # if > 1M combinations, skip
        return fit_w1, fit_w2

    for indices in combinations(range(n), k):
        if len(w1) > len(w2):
            a = np.array(list(w1))[list(indices)]
            b = np.array(list(w2))
        else:
            a = np.array(list(w1))
            b = np.array(list(w2))[list(indices)]

        score = np.sum(a == b)
        if score > max_score:
            fit_w1 = list(a)
            fit_w2 = list(b)
            max_score = score

    return fit_w1, fit_w2

def get_confusion_matrix(word_gts, word_preds, classes):
    """
    Arguments:
        word_gts (array of strings)
        word_preds (array of strings)
        classes (list)
    """
    null = ' '
    classes_new = classes
    if classes_new[-1] is None:
        classes_new[-1] = null
    classes = classes_new

    m = np.zeros((len(classes),len(classes)), dtype='int32')
    for gt, pred in zip(word_gts, word_preds):
        if len(pred) == 0 or len(gt) == 0:
            continue
        gt, pred = get_best_fit(gt, pred, classes, null=null)
        m = m + sklearn.metrics.confusion_matrix(gt, pred,
                labels=classes).astype('int32')

    return m

def parse_task3_data(gt_path, pred_path):
    with open(gt_path, 'r') as f:
        gt_file = f.read()
    with open(pred_path, 'r') as f:
        pred_file = f.read()

    pred_pattern = r'(\d+)\.png,\ "(.*)"\n'
    gt_pattern = r'word_(\d+)\.png,\ "(.*)"\n'

    pred_pairs = re.findall(pred_pattern, pred_file)
    gt_pairs = re.findall(gt_pattern, gt_file)

    pred_dict = dict(pred_pairs)
    gt_dict = dict(gt_pairs)

    gts, preds = [], []
    for key in sorted(pred_dict.keys()):
        gts.append(gt_dict[key])
        preds.append(pred_dict[key])

    return gts, preds

def export_confusion_matrix(gt_path, pred_path):
    gts, preds = parse_task3_data(gt_path, pred_path)
    m = get_confusion_matrix(gts, preds, datasets.ALPHANUMERIC)

    df = pd.DataFrame(m).to_csv('confusion_matrix.csv', header=None,
            index=None)

    return m
