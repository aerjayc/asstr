import numpy as np
import torch
import torchvision.transforms.functional as F

import cv2
import PIL.Image
import pandas as pd
import os.path
from pathlib import Path

import datasets
from image_proc import genCharBB, cropBB
from dataset_utils import filter_BBs, expand_BBs, onehot_to_chars,\
                          get_filenames
from character_grouper import character_grouper, words_to_wordBBs

from classifier import CharClassifier
from craft.craft import CRAFT


def map_to_crop(img, char_map, char_size=(64,64), cuda=True, expand_factor=3,
                thresh=0.5):
    if isinstance(img, PIL.Image.Image):
        C, W, H = 3, img.width, img.height
    if isinstance(img, (np.ndarray, torch.Tensor)):
        if img.shape[0] == 3:
            C, H, W = img.shape
            if isinstance(img, np.ndarray):
                img = img.transpose(1,2,0)
            else:   # if torch.Tensor
                img = img.permute(1,2,0).cpu().detach().numpy()
        else:
            H, W, C = img.shape

    if isinstance(char_map, torch.Tensor):
        char_map = char_map.cpu().detach().numpy()

    # generate character bounding boxes from heatmap
    charBBs = genCharBB(char_map, xywh=False, thresh=thresh) * 2

    # expand
    charBBs = expand_BBs(charBBs, (W,H), factor=2)

    # filter faulty values
    charBBs, _ = filter_BBs(charBBs, None, (W,H), verbose=True)

    # sort
    lines, words, words_idx = character_grouper(charBBs)
    if len(words) == 0:     # if no word detected
        return None, None, None, None

    charBBs = np.concatenate(words)             # sorted
    wordBBs = words_to_wordBBs(words)

    # expand
    charBBs = expand_BBs(charBBs, (W,H), factor=expand_factor/2)

    # filter faulty values
    charBBs, _ = filter_BBs(charBBs, None, (W,H), verbose=True)

    # get the cropped chars
    N = len(charBBs)
    cropped_chars = np.zeros((N, C, *char_size))
    for i, charBB in enumerate(charBBs):
        # crop + convert to numpy (H, W, C)
        cropped = cropBB(img, charBB, fast=True).astype('float32')

        # resize
        cropped = cv2.resize(cropped, dsize=char_size)

        cropped_chars[i] = cropped.transpose(2,0,1) # CHW

    cropped_chars = torch.from_numpy(cropped_chars)

    if cuda:
        cropped_chars = cropped_chars.cuda()

    return cropped_chars, charBBs, wordBBs, words_idx


def recognizer(img, detector_model, classifier_model, alphabet,
               show=False, expand_factor=3, thresh=0.5):
    """
    Arguments:
        - img (torch.Tensor): shaped (C, H, W)
    """
    img = img[None,...].cuda()

    # get heatmaps
    with torch.no_grad():
        output, _ = detector_model(img.float())
        char_heatmap = output[0, :, :, :1]  # (H, W, 1)

    cropped_chars, charBBs, wordBBs, words_idx = map_to_crop(img[0],
                                                 char_heatmap, thresh=thresh,
                                                 expand_factor=expand_factor)
    if charBBs is None:
        return None, None

    # get character predicitons
    with torch.no_grad():
        onehots = classifier_model(cropped_chars.double())
        chars = onehot_to_chars(onehots, alphabet)

    # string has same order as charBBs
    string = "".join(['?' if c == None else c for c in chars])

    # words has same order as wordBBs
    words = []
    for indices in words_idx:
        word = "".join([string[i] for i in indices])
        words.append(word)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,10), dpi=80)
        boxed = (img[0].cpu().permute(1,2,0).numpy() + 1) / 2
        for (tl, tr, br, bl), word in zip(wordBBs, words):
            boxed = cv2.rectangle(boxed, tuple(tl), tuple(br), (0,255,0), 3)
            boxed = cv2.putText(boxed, word, tuple(tl), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255,0,0), 2)
        # for (tl, tr, br, bl) in charBBs:
        #     boxed = cv2.rectangle(boxed, tuple(tl), tuple(br), (0,0,255), 3)
        plt.imshow(boxed)
        plt.show()

    return wordBBs, words


def export_task1_4(test_img_dir, test_gt_dir,
                     classifier_weight_path, detector_weight_path,
                     alphabet=datasets.ALPHANUMERIC, submit_dir='submit',
                     thresh=0.3):
    testset = datasets.ICDAR2013TestDataset(test_gt_dir, test_img_dir)
    max_size = (700, 700)

    # make submit dirs
    submit_task1_dirpath = os.path.join(submit_dir, 'text_localization_output')
    submit_task4_dirpath = os.path.join(submit_dir, 'end_to_end_output')
    Path(submit_task1_dirpath).mkdir(parents=True, exist_ok=True)
    Path(submit_task4_dirpath).mkdir(parents=True, exist_ok=True)

    # instantiate models
    detector = CRAFT(pretrained=True, num_class=2, linear=True).cuda()
    detector.load_state_dict(torch.load(detector_weight_path))
    detector.eval()

    classifier = CharClassifier(num_classes=len(alphabet)).double().cuda()
    classifier.load_state_dict(torch.load(classifier_weight_path))
    classifier.eval()


    # get predictions for all test images
    for img, _, _, pil_img in testset:
        filename = os.path.basename(pil_img.filename)
        orig_w, orig_h = pil_img.width, pil_img.height

        # resize when # of pixels exceeds size
        if (max_size is not None) and ((orig_w*orig_h) > (max_size[0]*max_size[1])):
            """
            max_w, max_h = max_size
            if orig_w > orig_h:
                ratio = max_w / orig_w
            elif orig_h > orig_w:
                ratio = max_h / orig_h

            new_w, new_h = int(ratio*orig_w), int(ratio*orig_h)
            print(f"Resizing ({orig_w}, {orig_h}) to ({new_w}, {new_h})",
                    end='')
            img = img.resize((new_w, new_h))
            """
            img = img.permute(1,2,0).cpu().numpy()
            img = cv2.resize(img, dsize=max_size)  # HWC
            img = torch.from_numpy(img).permute(2,0,1)  # CHW
            new_w, new_h = max_size

        print(filename + '... ', end='')
        wordBBs, words = recognizer(img, detector, classifier, alphabet,
                                    thresh=thresh)
        if wordBBs is None:    # write blank files
            print("No characters found. Making blank file... ", end='')
            submit_name = f"res_{filename[:-4]}.txt"
            submit_path = os.path.join(submit_task1_dirpath, submit_name)
            with open(submit_path, 'w') as f:
                f.write("")

            submit_path = os.path.join(submit_task4_dirpath, submit_name)
            with open(submit_path, 'w') as f:
                f.write("")

            continue

        unsquish_factor = np.array([orig_w/new_w, orig_h/new_h]).reshape(1,1,2)
        wordBBs = unsquish_factor * wordBBs

        print('Formatting submission... ', end='')
        # create submission contents for task 1 (text localization)
        xymin = np.min(wordBBs, axis=1).astype('int32')
        xymax = np.max(wordBBs, axis=1).astype('int32')
        xmin, ymin = xymin[:,0], xymin[:,1]
        xmax, ymax = xymax[:,0], xymax[:,1]

        contents = {'xmin': xmin, 'ymin': ymin,
                    'xmax': xmax, 'ymax': ymax}
        submission = pd.DataFrame(contents)

        # write to file
        submit_name = f"res_{filename[:-4]}.txt"
        submit_path = os.path.join(submit_task1_dirpath, submit_name)
        submission.to_csv(submit_path, header=False, index=False)

        # create submission contents for task 4 (end to end recognition)
        x1, y1 = xmin, ymin
        x2, y2 = xmax, ymin
        x3, y3 = xmax, ymax
        x4, y4 = xmin, ymax
        contents = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'x3': x3, 'y3': y3, 'x4': x4, 'y4': y4,
                    'transcription': words}
        submission = pd.DataFrame(contents)
        submit_path = os.path.join(submit_task4_dirpath, submit_name)
        submission.to_csv(submit_path, header=False, index=False)

        print(f'Done.')

    print("Done!")


def export_task3(img_dir, classifier_weight_path, detector_weight_path,
                 alphabet=datasets.ALPHANUMERIC, submit_dir='submit',
                 thresh=0.3, verbose=True, gt_path=None):
    # make submit dirs
    Path(submit_dir).mkdir(parents=True, exist_ok=True)

    # instantiate models
    detector = CRAFT(pretrained=True, num_class=2, linear=True).cuda()
    detector.load_state_dict(torch.load(detector_weight_path))
    detector.eval()

    classifier = CharClassifier(num_classes=len(alphabet)).double().cuda()
    classifier.load_state_dict(torch.load(classifier_weight_path))
    classifier.eval()

    # prepare results filename
    submit_name = f"task3.txt"
    submit_path = os.path.join(submit_dir, submit_name)

    # read gt file
    if gt_dir:
        with open(gt_dir, 'a') as f:
            gt = f.read()
    else:
        gt = ""

    # get predictions for all images
    img_names = get_filenames(img_dir, ['.png'], recursive=False)
    for img_name in img_names:
        res_name = img_name[4:]
        img_path = os.path.join(img_dir, img_path)

        # convert PIL.Image to torch.Tensor (1, C, H, W)
        img = np.array(PIL.Image.open(img_path))
        img = torch.from_numpy(img).transpose(2,0,1)[None, ...].cuda()

        # get heatmaps
        with torch.no_grad():
            output, _ = detector_model(img.float())
            char_heatmap = output[0, :, :, :1]  # (H, W, 1)

        cropped_chars, _, _, _ = map_to_crop(img[0],
                                             char_heatmap, thresh=thresh,
                                             expand_factor=expand_factor)
        if cropped_chars is not None:
            # get character predicitons
            with torch.no_grad():
                onehots = classifier_model(cropped_chars.double())
                chars = onehot_to_chars(onehots, alphabet)

            # string has same order as charBBs
            string = ""
            for c in chars:
                if c is None:
                    string += '?'
                elif c == '"':
                    string += r'\"'
                elif c == '\\':
                    string += r'\\'
                else:
                    string += c
        else:
            string = ""

        # write to file
        contents = f'{res_name}, "{string}"'
        if verbose:
            print(contents, end='')
            if gt:
                pattern = re.compile(re.escape(img_name) + r',\ "(.*)"')
                matches = re.search(pattern, gt)
                if matches:
                    print("\tgt: {matches.group(1)}", end='')
            print('')

        with open(submit_path, 'a') as f:
            f.write(contents, end='')

    print("Done!")

