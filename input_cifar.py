from utils import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import torchvision.datasets as ds
import torchvision.transforms.functional as ttf
from einops import repeat
import os.path as osp
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser("ff encoding")
    parser.add_argument("--cifar_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, help='assoc or effects')
    return parser.parse_args()

args = parse_args()

cif = ds.CIFAR100(args.cifar_path, download=True)

num_kernel = 64
num_row = 8
num_col = 8
H, W = 32, 32
# overlap = 4
filter_size = [9, 9]
num_stimuli = 5


def add_noise(cif, num_imgs, noise_level, std_threshold=0.2, iidx=None):
    """
    imgs: list of tensor with shape (C, H, W)
    """
    # construct noise corpus
    h, w = cif[0][0].size
    idx_all = np.arange(h*w)
    idx_corrupt = [np.random.choice(idx_all, int(len(idx_all) * noise_level), False) for _ in range(10)]
    corrupt_patterns = [torch.rand(len(idx_corrupt[i])) for i in range(len(idx_corrupt))]

    imgs = []
    imgs_noise = []

    if iidx is None:
        iidx = np.random.choice(len(cif), num_imgs)

    for i in iidx:
        img = ttf.to_tensor(cif[i][0])
        img = ttf.rgb_to_grayscale(img).movedim(0, -1)
        # check std
        # if torch.std(img, unbiased=True) <= std_threshold:
        #     continue
        imgs.append(img)
        # noise corruption
        samples = [img]
        for idx, pattern in zip(idx_corrupt, corrupt_patterns):
            noise_img = img.clone().flatten()
            noise_img[idx] = pattern
            ni = noise_img.clone().reshape(*img.shape)
            samples.append(ni)
        imgs_noise.append(torch.stack(samples))

    return torch.stack(imgs).detach().numpy().squeeze(), torch.stack(imgs_noise).detach().numpy().squeeze()


def ff_encode(X):
    filters = np.load("filter_256.npy")
    # 64 filters used for experiments
    filter_list = [113, 233,  51,  28,  64,  62,  56, 252, 229, 125, 213,  23,  
                   90, 91, 127,  63, 176,  75,   8,  97, 106, 162, 105, 247, 204,  
                   58, 87,  82,   1, 108,  96, 250,  76,  69, 216, 217,  81, 211, 
                   160,3, 155, 147, 169, 199,  17, 207, 215, 232,  86,  84,  72, 
                   152, 218, 136, 178, 158,  10,  48, 139, 187, 230, 253, 180, 146]
    
    response = []
    imgs = X.reshape(-1, H, W, 1)
    for i, n in enumerate(filters[filter_list]):
        kernel = n.reshape(9,9,1,1)
        # tmp = (tf.nn.conv2d(imgs, kernel, strides=[1, 4, 4, 1], padding='VALID').numpy()).reshape(num_stimuli, num_row, num_col)
        tmp = tf.nn.conv2d(imgs, kernel, strides=[1, 3, 3, 1], padding='VALID').numpy()
        tmp = tmp.squeeze()
        response.append(tmp)
    response = np.stack(response).transpose(1, 0, 2, 3)
    r_in = np.abs(response)
    
    return r_in


if args.task == 'effects':
    # output dir
    fam_out_dir = 'input_fameff'
    Path(fam_out_dir).mkdir(parents=True, exist_ok=True)

    count, i = 0, 0
    imgs = []
    while True:
        img, label = cif[i]
        i += 1
        img = ttf.to_tensor(img)
        img = ttf.rgb_to_grayscale(img).movedim(0, -1)
        imgs.append(img)
        if i >= 500:
            break
    X = torch.stack(imgs).squeeze().detach().cpu().numpy()  # 100, 32, 32
    np.save(osp.join(fam_out_dir, 'X_cifar500_pure.npy'), X)

    # ff encoding
    r_in = ff_encode(X)
    np.save(osp.join(fam_out_dir, 'r_in_cifarPure_abs_500_64_8_8.npy'), r_in)

elif args.task == 'assoc':
    # output dir
    noise_out_dir = 'input_famassoc'
    Path(noise_out_dir).mkdir(parents=True, exist_ok=True)

    ## clear-noise images pairs for familiarity association experiment
    idx = [4, 7, 3045, 11739, 32045]  # cifar image index

    _, X_10 = add_noise(cif, num_stimuli, noise_level=0.1, iidx=idx) 
    _, X_30 = add_noise(cif, num_stimuli, noise_level=0.3, iidx=idx)
    _, X_50 = add_noise(cif, num_stimuli, noise_level=0.5, iidx=idx)

    X_clear = repeat(X_10[:, 0], 'n h w -> n p h w', p=10)
    X_10_n = X_10[:, 1:]
    X_30_n = X_30[:, 1:]
    X_50_n = X_50[:, 1:]
    X = np.stack([X_clear, X_10_n, X_30_n, X_50_n], 1)
    # save
    np.save(osp.join(noise_out_dir, 'X_cifar5_all_noise.npy'), X)

    # ff encoding
    r_in = ff_encode(X)
    r_in = rearrange(r_in, "(n l p) k h w -> n l p k h w", n=num_stimuli, l=4)
    np.save(osp.join(noise_out_dir, 'r_in_cifar_all_noise_abs_5_4_10_64_8_8.npy'), r_in)
