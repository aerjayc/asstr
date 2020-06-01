import os.path
from pathlib import Path
import time

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import image_proc
import train_detector

from datasets import SynthCharDataset, ICDAR2013Dataset, ICDAR2013CharDataset
from classifier import CharClassifier

ALPHABET = list("AaBbCDdEeFfGgHhIiJjKLlMmNnOPQqRrSTtUVWXYZ") + [None,]


def train_loop(dataloader, model, optimizer, criterion, epochs, weight_dir,
               valloader=None, cuda=True, per_epoch=True, T_save=1, T_print=1,
               supress_errors=True):
    T_start = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        try:
            for i, (inputs, labels) in enumerate(dataloader):
                if len(labels) == 0:
                    # print("Skipping batch with 0 length")
                    continue
                if cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # free up space
                del inputs
                del outputs
                del labels
                torch.cuda.empty_cache()

                if per_epoch:
                    running_loss += loss.item()
                else:
                    # print statistics on a per batch basis
                    running_loss = train_detector.print_statistics(
                                    running_loss, loss, i, epoch,
                                    model=model, weight_dir=weight_dir,
                                    T_save=T_save, T_print=T_print,
                                    T_start=T_start)

            if not per_epoch:
                continue

            # check validation set
            if valloader:
                pass

            # print statistics every T_print epochs
            if epoch % T_print == T_print-1:
                print('%d)\tloss: %.5f' % (epoch + 1, running_loss))

            # save every T_save epochs
            if weight_dir and (epoch % T_save == T_save-1):
                weight_fname = f'w_{epoch}.pth'
                weight_path = os.path.join(weight_dir, weight_fname)

                print(f'\nSaving at {epoch+1}-th epoch')
                torch.save(model.state_dict(), weight_path)
                T_end = time.time()
                print(f'Elapsed time: {T_end-T_start}\n')

            # stopping criterion
        except KeyboardInterrupt:
            weight_fname = f'w_{epoch}_interrupt.pth'
            weight_path = os.path.join(weight_dir, weight_fname)

            print(f'\nSaving at {weight_path}')
            torch.save(model.state_dict(), weight_path)
            T_end = time.time()
            print(f'Elapsed time: {T_end-T_start}\n')

            break
        except Exception as e:
            print(e)
            try:
                print("img.shape =", img.shape)
                print("gt.shape =", gt.shape)
            except:
                pass
            if supress_errors:
                continue
            else:
                break


    T_end = time.time()
    print(f'Finished Training ({T_end-T_start} secs).')


def synthetic_classifier_training():
    home = False
    if home:
        windows_path_prefix = "C:"
        linux_path_prefix = "/mnt/A4B04DFEB04DD806"

        path_prefix = linux_path_prefix
        img_dir = path_prefix + '/Users/Aerjay/Downloads/SynthText/SynthText'
        gt_path = path_prefix + '/Users/Aerjay/Downloads/SynthText/gt_v7.3.mat'
        weight_dir = '/home/aerjay/Documents/thesis/weights'
    else:
        gt_path = "/home/eee198/Downloads/SynthText/gt_v7.3.mat"
        img_dir = "/home/eee198/Downloads/SynthText/images"

        weight_folder = 'classifier/synth'
        weight_dir = "/home/eee198/Downloads/SynthText/weights/" + weight_folder
        # weight_fname = None     # pretrained weights

    # make weight_dir if it doesn't exist
    Path(weight_dir).mkdir(parents=True, exist_ok=True)

    cuda = False
    size = (64,64)

    epochs = 1

    dataset = SynthCharDataset(gt_path, img_dir, size)

    N = len(dataset)
    train_test_val = [int(0.8*N), int(0.15*N)]
    train_test_val += [N - sum(train_test_val),]
    train, test, validation = random_split(dataset, train_test_val)

    trainloader = DataLoader(train, batch_size=1, shuffle=True,
                                collate_fn=SynthCharDataset.collate_fn)
    # valloader = DataLoader(validation, batch_size=1, shuffle=True,
                                # collate_fn=SynthCharDataset.collate_fn)

    model = CharClassifier(num_classes=len(dataset.alphabet)).double()
    if cuda:
        model = model.cuda()

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #

    train_loop(trainloader, model, optimizer, criterion, weight_dir,
               epochs=epochs, per_epoch=False, T_print=100, T_save=10000)


if __name__ == '__main__':
    pass
    # classes = ['a', 'e', 'i', 'o', 'u']
    # simpleGenChar(10, img_dir, mat_dir, N_images=100)
    # genBalancedCharDataset(20, img_dir, mat_path, char_dir, classes=classes)

