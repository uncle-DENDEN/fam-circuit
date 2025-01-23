from utils import *
from model import ProtoRNN

import scipy.linalg as linalg
import numpy as np
import torch
from einops import rearrange, repeat
import argparse
from pathlib import Path


def get_my_epoch(input_string, e):
    epoch_pos = input_string.find('epoch')
    end_pos = epoch_pos + 5  # Start from the character after 'epoch'
    
    while end_pos < len(input_string) and input_string[end_pos].isdigit():
        end_pos += 1

    # Replace the number after 'epoch' with e
    new_string = input_string[:epoch_pos + 5] + str(e) + input_string[end_pos:]

    return new_string


def find_eigenmode(epoch, img, level, pattern, paths):
    
    # params
    num_imgs = 5
    num_patterns = 10
    mix_coef = 2
    device = torch.device('cpu')
    postfix = {0: 'clear', 1: '10%', 2: '30%', 3: '50%'}

    # get input and mixing index
    r_in = np.load(paths.input_path)
    r_in, t_ind, n_ind = insertTeachingSig(r_in, p=mix_coef)

    # get model
    num_kernel = 64
    num_row = 8
    num_col = 8
    tau_stim = 150
    delta_t = 1 
    save_every_stim = 15

    f = ProtoRNN(num_kernel, num_row, num_col, tau_stim, delta_t, save_every_stim)
    f.to(device)

    # static weights
    wei = f._wei0.cpu().numpy()
    wie = f._wie0.cpu().numpy()

    # get tau_inverse
    tau_inv_flat = np.concatenate([np.array([1/f.tau_ue]*f.N_e), np.array([1/f.tau_ui]*f.N_i)])  # Ne+Ni
    tau_inv = np.diag(tau_inv_flat)

    # get weight matrix ----------------------------------------------------------------------------
    if epoch != 0:    
        weight_path_e = get_my_epoch(paths.weight_path, epoch-1)
        wee = np.load(weight_path_e)
    else:
        wee = f.wrp0.cpu().numpy()
    # -------------------------------------------------------------------------------------------------------------------
    w = np.block([
        [wee, -wei],
        [wie, np.zeros_like(wee)]
    ])

    # got equilibrium point ----------------------------------------------------------------------------
    if epoch != 0:
        response_path_e = get_my_epoch(paths.response_path, epoch-1)
    else:
        response_path_e = paths.response_path_pre
    u = np.load(response_path_e)
    # -------------------------------------------------------------------------------------------------------------------
    u = seqUnmix(u, t_ind, n_ind, n_imgs=num_imgs, npa=num_patterns)  # n_img*nl*np, 16, Ne+Ni
    u_eq = u[:, -2:].mean(1)  # n_img*nl*np, Ne+Ni

    # get phi_prime
    dphi_dreq = np.maximum(0, 2*u_eq)  # n_img*nl*np, Ne+Ni
    phi_prime_flat = dphi_dreq * np.expand_dims(tau_inv_flat, 0)  # n_img*nl*np, Ne+Ni
    phi_prime_flat = rearrange(phi_prime_flat, '(n l p) a -> n l p a', n=num_imgs, p=num_patterns)  # n_img, nl, np, Ne+Ni
    del dphi_dreq

    # get corresponding sensitivity matrix
    phi_prime_mn = np.diag(phi_prime_flat[img, level, pattern])  # Ne+Ni, Ne+Ni
    
    # get Jacobian for the single img
    jmn = phi_prime_mn @ w - tau_inv

    # eigendecomposition
    vals, vl, vr = linalg.eig(jmn, left=True)

    #  -----------------------------------------------------------
    if epoch != 0:
        print(f'\t[INFO] Decomposition of Jacobian finished for pattern{pattern}, image{img}, epoch{epoch}, {postfix[level]}')
    else:
        print(f'\t[INFO] Decomposition of Jacobian finished for pattern{pattern}, image{img}, pre, {postfix[level]}')
    # ---------------------------------------------------------------------------------

    vals = np.real(vals)
    vl = np.real(vl)
    vr = np.real(vr)
    vl_prime = phi_prime_mn @ vl
    
    # sort 
    sortind = np.argsort(vals)[::-1]
    vals_sorted = vals[sortind]
    vl_sorted = vl[:, sortind]
    vr_sorted = vr[:, sortind]
    vl_prime_sorted = vl_prime[:, sortind]

    del phi_prime_flat, w
    
    return vals_sorted, vl_sorted, vr_sorted, vl_prime_sorted

