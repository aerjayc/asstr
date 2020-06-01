import os.path
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from craft.craft import CRAFT

import time

def show_samples(imgs, i=None, feature_type="img", title=None, channel=None):
    imgs = imgs.detach().cpu().numpy()
    if i == None:
        img = imgs
    else:
        img = imgs[i]

    if feature_type == "img":
        img = img.transpose(1,2,0)
    elif feature_type == "gt":
        pass
    else:
        print(f"Warning: feature_type should be 'img' or 'gt', not " +
              f" '{feature_type}'. Assuming 'gt'.")

    if channel != None:
        img = img[:,:,channel]

    if title is None:
        title = f"img[{i}]"

    plt.figure(figsize=(10,10))
    plt.title(title)
    plt.imshow(img, interpolation='nearest')

    plt.show()


def print_statistics(running_loss, loss, i, epoch, model=None,
                     T_start=None, T_print=100, T_save=10000, weight_dir=None,
                     weight_fname_template=None, weight_fname_args=None,
                     print_template=None, print_args=None):
    running_loss += loss.item()

    # print every T_print minibatches
    if i % T_print == T_print-1:
        if print_template is None:
            print_template = '[%d, %5d] loss: %f'
            print_args = (epoch + 1, i + 1, running_loss/T_print)
        print(print_template % print_args)
        running_loss = 0.0

    # save every T_save minibatches
    if i % T_save == T_save-1:
        if weight_dir is not None:
            if weight_fname_template is None:
                weight_fname_template = "w_%d.pth"
                weight_fname_args = (i+1,)

            weight_fname = weight_fname_template % weight_fname_args
            weight_path = os.path.join(weight_dir, weight_fname)

            print(f"\nsaving at {i+1}-th batch")
            torch.save(model.state_dict(), weight_path)
            T_end = time.time()
            if T_start is not None:
                print(f"\nElapsed time: {T_end-T_start}")

    return running_loss

def step(model, criterion, optimizer, input, target):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    output,_ = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    return loss, optimizer

def init_data(gt_path, img_dir, dataset, dataset_kwargs={},
              dataloader_kwargs={}, testloader_kwargs={}, **kwargs):
    # defaults:
    dataset_defaults = {}
    dataloader_defaults = {
        "batch_size": 4,
        "shuffle": True
    }
    testloader_defaults = {
        "batch_size": 4,
        "shuffle": False
    }
    # if size is set, no need to use custom collate_fn
    if ("size" not in dataset_kwargs) or (dataset_kwargs["size"] is None):
        dataloader_defaults["collate_fn"] = SynthCharMapDataset.collate_fn

    for entry in dataset_defaults:
        if entry not in dataset_kwargs:
            dataset_kwargs[entry] = dataset_defaults[entry]
    for entry in dataloader_defaults:
        if entry not in dataloader_kwargs:
            dataloader_kwargs[entry] = dataloader_defaults[entry]

    # remember requires_grad=True
    dataset = SynthCharMapDataset(gt_path, img_dir, **dataset_kwargs)
    if "split" in kwargs:
        trainset, testset = torch.utils.data.random_split(dataset,
                                                          kwargs["split"])
        testloader = DataLoader(testset, **testloader_kwargs)
    else:
        trainset = dataset
        testset = None
        testloader = None

    trainloader = DataLoader(trainset, **dataloader_kwargs)

    return trainloader, trainset, testloader, testset

def init_model(weight_dir, weight_path=None, num_class=2, linear=True):
    # make weight_dir if it doesn't exist
    Path(weight_dir).mkdir(parents=True, exist_ok=True)

    # input: NCHW
    model = CRAFT(pretrained=True, num_class=num_class, linear=linear).cuda()
    # output: NHWC

    if weight_path:
        # pretrained_weight_path = os.path.join(weight_dir, weight_fname)
        model.load_state_dict(torch.load(weight_path))
        model.eval()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001) # tweak parameters

    return model, criterion, optimizer

def train_ic13_loop(trainloader, model, criterion, optimizer,
               epochs=100, epoch_start=0, T_print=200, T_save=10):
    T_start = time.time()
    for epoch in range(epoch_start, epochs):
        running_loss = 0.0
        running_loss_mini = 0.0
        for i, (inputs, targets) in enumerate(trainloader):
            loss, optimizer = step(model, criterion, optimizer, inputs, targets)

            running_loss += loss.item()
            running_loss_mini += loss.item()

            if i % T_print == T_print-1:
                T_end = time.time()
                print('\tbatch %3d\tloss: %0.5f' % (i+1, running_loss/T_print), end='\t')
                print(T_end-T_start, 'secs elapsed')
                running_loss_mini = 0.0

        # print statistics
        T_end = time.time()
        print('epoch %3d\tloss: %f' % (epoch + 1, running_loss))
        print(T_end-T_start, 'secs elapsed')

        # save
        if epoch % T_save == T_save-1:
            save_model(model, weights_dir, weight_fname)

    print("Done!")


def train_loop(dataloader, model, criterion, optimizer, weight_dir, epochs=1):
    T_start = time.time()
    for epoch in range(epochs):
        running_loss, hard_running_loss = 0.0, 0.0

        while True:
            try:
                for i, data in enumerate(dataloader):
                    # print(i, end='')

                    # unpack data:
                    if len(data) == 2:
                        img, target = data
                    else:
                        img, target, hard_img, hard_target = data

                    loss, optimizer = step(model, criterion, optimizer, img, target)
                    running_loss = print_statistics(running_loss, loss, i,
                                        epoch, model=model, T_start=T_start,
                                        weight_dir=weight_dir)

                    if len(data) == 2:
                        continue
                    hard_img, hard_target = hard_img[0], hard_target[0]

                    loss, optimizer = step(model, criterion, optimizer, hard_img, hard_target)
                    hard_running_loss = print_statistics(hard_running_loss, loss,
                                                i, epoch, T_start=T_start)
                break
            except KeyboardInterrupt:
                weight_fname_interrupt = f"w_{i+1}_interrupt.pth"
                weight_path_interrupt = os.path.join(weight_dir, weight_fname_interrupt)

                print(f"\nSaving at {i+1}-th batch...")
                torch.save(model.state_dict(), weight_path_interrupt)

                T_end = time.time()
                print(f"Elapsed time: {T_end-T_start}")
                break
            except Exception as err:
                print("")
                print(err)
            else:
                print(f"Occured at i={i}, ")
                print(f"img.shape = {img.shape}, target.shape = {target.shape}")
                print(f"hard_img.shape = {hard_img.shape}, hard_target.shape ="
                        + f"{hard_target.shape}")

    print("Finished training.")

    T_end = time.time()
    print(f"\nTotal elapsed time: {T_end-T_start}")

    return img, target

def train_synthetic(weight_folder):
    gt_path = "/home/eee198/Downloads/SynthText/gt_v7.3.mat"
    img_dir = "/home/eee198/Downloads/SynthText/images"
    weight_dir = "/home/eee198/Downloads/SynthText/weights/" + weight_folder
    weight_path = None     # pretrained weights

    dataset_kwargs = {
        "cuda": True
    }
    dataloader_kwargs = {
        "batch_size": 4
    }
    num_class = 2
    epochs = 1

    trainloader, _, _, _ = init_data(gt_path, img_dir,
                                     dataset=SynthCharMapDataset,
                                     dataset_kwargs=dataset_kwargs,
                                     dataloader_kwargs=dataloader_kwargs)
    model, criterion, optimizer = init_model(weight_dir, weight_path, num_class)

    train_loop(trainloader, model, criterion, optimizer,
               weight_dir, epochs=epochs)

def train_ic13(weight_folder):
    img_dir = "/home/eee198/Downloads/icdar-2013/train_images"
    gt_dir = "/home/eee198/Downloads/icdar-2013/train_char_gt"
    weight_dir = "/home/eee198/Downloads/weights/detector/"
    weight_path = "synth_pretrained.pth"

    dataset_kwargs = {"cuda": True}
    dataloader_kwargs = {"batch_size": 4}
    num_class = 2
    epochs = 100

    trainloader,_,testloader,_ = init_data(gt_dir, img_dir,
                                           dataset=ICDAR2013MapDataset,
                                           dataset_kwargs=dataset_kwargs,
                                           dataloader_kwargs=dataloader_kwargs)
    model, criterion, optimizer = init_model(weight_dir, weight_path, num_class)

    train_loop(trainloader, model, criterion, optimizer,
               weight_dir, epochs=epochs)


if __name__ == '__main__':
    pass
