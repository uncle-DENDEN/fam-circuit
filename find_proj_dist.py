from utils import *
from model import ProtoRNN
from find_eigenmode import find_eigenmode

from numpy.linalg import norm
import scipy.linalg as linalg
from sklearn.decomposition import PCA
import numpy as np
from pathlib import Path
import joblib
import os
import re
import argparse

import torch
from einops import rearrange, repeat


def str_int(value):
    try:
        return int(value)
    except ValueError:
        return str(value)
    

# params
def parse_args():
    parser = argparse.ArgumentParser("find_proj_stats")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--response_path", type=str, required=True)
    parser.add_argument("--response_path_pre", type=str, required=True)
    parser.add_argument("--save_jac", nargs='+', type=str_int, required=True, 
                        help="image sample where jacbian matrix is saved. Should be [img, level, sample] e.g. [1, 'clear', 0]")

    return parser.parse_args()

args = parse_args()

patterns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
postfix = {0: 'clear', 1: '10%', 2: '30%', 3: '50%'}
num_imgs = 5
modes_ns = [200, 200, 200, 200, 200]

## Model
f = ProtoRNN(64, 8, 8, 150, 1, 75, wie=30)
f.to(torch.device("cpu"))

# get t_ind for unmixing
r_in = np.load(args.input_path)
r_in, t_ind, n_ind = insertTeachingSig(r_in, p=2)


def get_my_epoch(input_string, e):
    epoch_pos = input_string.find('epoch')
    end_pos = epoch_pos + 5  # Start from the character after 'epoch'
    
    while end_pos < len(input_string) and input_string[end_pos].isdigit():
        end_pos += 1

    # Replace the number after 'epoch' with e
    new_string = input_string[:epoch_pos + 5] + str(e) + input_string[end_pos:]

    return new_string

def get_my_epoch_img_nl(path, img, nl, epoch=None):
    # Replace the number after 'epoch'
    if epoch is not None:
        path = re.sub(r'(epoch)\d+', rf'\g<1>{epoch}', path)
    
    # Replace the number after 'img'
    path = re.sub(r'(img)\d+', rf'\g<1>{img}', path)
    
    # Replace the percentage part
    path = re.sub(r'\d+%', f'{nl}', path)
    
    return path


def get_normalized_dist(r, img, delta):
    """
    Get the distance (estimate of derivatives). r should be a list of array, each of shape (n_modes, num_img*2), containing projected 
    stationary state of all images in 2 adjacent noise level. The linearization is performed around the first level. Derivative is 
    approximated by the distance between 2 projected stationary states/inputs of the specified image, divided by increment of noise level 
    (10% as a unit). The estimated derivative is normalized by the overall distance scale of the projected space, which is the 
    variance of projected stationary states/inputs of all images at the linearized noise level. 
    """
    r = rearrange(r, 'k (n l) -> n l k', l=2)
    r_img = r[img]  # l, k
    d = norm((r_img[0] - r_img[1]) / delta)

    r_scale = r[:, 0]  # n, k
    r_scale_ref = r_scale[[img]]
    cov_scale = (r_scale - r_scale_ref).T @ (r_scale - r_scale_ref) / (num_imgs - 1)
    scale = np.sqrt(np.trace(cov_scale))

    d_norm = d / scale

    return d, scale, d_norm

def analyze_eigenmodes_pre_v3(img, pp, num_modes=None):
    '''
    Better derivative measures 10%-0% at 0%, 30%-10% at 10%, 50%-30% at 30%. 
    '''
    ff = np.load(args.input_path) 
    ff = rearrange(ff, 'n l p k h w -> (n l p) (k h w)')  # n_img*nl*np, Ne

    u = np.load(args.response_path_pre) 
    u = seqUnmix(u, t_ind, n_ind, n_imgs=num_imgs, npa=10)  # n_img*nl*np, 16, Ne+Ni
    r = f.r_numpy(u)  # n_img*nl*np, 16, Ne+Ni
    r_eq = r[:, -2:].mean(-2)  # n_img*nl*np, Ne+Ni

    r_eq = rearrange(r_eq, '(n l p) k -> n l p k', n=num_imgs, l=4)
    ff = rearrange(ff, '(n l p) k -> n l p k', n=num_imgs, l=4)

    # take the corresponding noise pattern where linearization performed
    r_eq = r_eq[:, :, pp]  # n, l, Ne+Ni
    ff = ff[:, :, pp]  # n, l, Ne

    d_o_pre_pi, scale_o_pre_pi, d_norm_o_pre_pi = np.empty((num_modes//4, 3)), np.empty((num_modes//4, 3)), np.empty((num_modes//4, 3))
    d_i_pre_pi, scale_i_pre_pi, d_norm_i_pre_pi = np.empty((num_modes//4, 3)), np.empty((num_modes//4, 3)), np.empty((num_modes//4, 3))
    
    # iterate over images
    for nl, delta in zip([0, 1, 2], [1, 2, 2]):

        val, vl, vr, vl_prime = find_eigenmode(0, img, nl, pp, args)

        if args.save_jac == [img, postfix[nl], pp]:
            data = {'val': val, "vl": vl, "vr": vr, "vl_prime": vl_prime}
            joblib.dump(
                data, 
                os.path.join('example_jac_matrix', f'jac_pre_img{img}_{postfix[nl]}_sample{pp}.pkl')
                )

        # get effective dimension
        vl_prime_norm = np.linalg.norm(vl_prime, axis=0)
        eff_dim = np.where(vl_prime_norm == 0)[0][0]
        num_modes_ = num_modes if num_modes is not None else eff_dim

        for nm_ in range(0, num_modes_, 4):
            N = nm_+1

            # slowest decaying mode
            mode_s = vl[:, :N]  # Ne+Ni, 20

            # change of variable modes
            mode_s_prime = vl_prime[:, :N]  # Ne+Ni, 20

            # projecting output of adjacent noise level to the mode
            r_eq_exc = rearrange(r_eq[:, [nl, nl+1]], 'n l k -> (n l) k')  # n*nl, Ne+Ni
            proj_output = mode_s.T @ r_eq_exc.T  # 20, t*nl

            # projecting input of adjacent noise level to the change of variable mode
            ff_ = rearrange(ff[:, [nl, nl+1], :], 'n l k -> (n l) k')  # n*nl, Ne
            proj_input = mode_s_prime[:f.N_e, :].T @ ff_.T  # 20, t*nl
                
            d_o_prepipnm, scale_o_prepipnm, d_norm_o_prepipnm = get_normalized_dist(proj_output, img, delta)
            d_i_prepipnm, scale_i_prepipnm, d_norm_i_prepipnm = get_normalized_dist(proj_input, img, delta)
            
            d_o_pre_pi[nm_//4, nl] = d_o_prepipnm
            scale_o_pre_pi[nm_//4, nl] = scale_o_prepipnm
            d_norm_o_pre_pi[nm_//4, nl] = d_norm_o_prepipnm
            d_i_pre_pi[nm_//4, nl] = d_i_prepipnm
            scale_i_pre_pi[nm_//4, nl] = scale_i_prepipnm
            d_norm_i_pre_pi[nm_//4, nl] = d_norm_i_prepipnm
    
    return d_o_pre_pi, scale_o_pre_pi, d_norm_o_pre_pi, d_i_pre_pi, scale_i_pre_pi, d_norm_i_pre_pi


def analyze_eigenmodes_trained_v3(epoch, img, pp, num_modes=None):
    '''
    Better derivative measures 10%-0% at 0%, 30%-10% at 10%, 50%-30% at 30%. 
    '''
    ff = np.load(args.input_path) 
    ff = rearrange(ff, 'n l p k h w -> (n l p) (k h w)')  # n_img*nl*np, Ne

    u_path = get_my_epoch(args.response_path, epoch-1)
    u = np.load(u_path)
    u = seqUnmix(u, t_ind, n_ind, n_imgs=num_imgs, npa=10)  # n_img*nl*np, 16, Ne+Ni
    r = f.r_numpy(u)  # n_img*nl*np, 76, Ne+Ni
    r_eq = r[:, -2:].mean(-2)  # n_img*nl*np, Ne+Ni

    r_eq = rearrange(r_eq, '(n l p) k -> n l p k', n=num_imgs, l=4)
    ff = rearrange(ff, '(n l p) k -> n l p k', n=num_imgs, l=4)
    
    # take the corresponding noise pattern where linearization performed
    r_eq = r_eq[:, :, pp]  # n, l, Ne+Ni
    ff = ff[:, :, pp]  # n, l, Ne

    d_o_pipe, scale_o_pipe, d_norm_o_pipe = np.empty((num_modes//4, 3)), np.empty((num_modes//4, 3)), np.empty((num_modes//4, 3))
    d_i_pipe, scale_i_pipe, d_norm_i_pipe = np.empty((num_modes//4, 3)), np.empty((num_modes//4, 3)), np.empty((num_modes//4, 3))

    # iterate over images
    for nl, delta in zip([0, 1, 2], [1, 2, 2]):

        val, vl, vr, vl_prime = find_eigenmode(epoch, img, nl, pp, args)

        if args.save_jac == [img, postfix[nl], pp]:
            data = {'val': val, "vl": vl, "vr": vr, "vl_prime": vl_prime}
            joblib.dump(
                data, 
                os.path.join('example_jac_matrix', f'jac_epoch{epoch}_img{img}_{postfix[nl]}_sample{pp}.pkl')
                )

        # get effective dimension
        vl_prime_norm = np.linalg.norm(vl_prime, axis=0)
        eff_dim = np.where(vl_prime_norm == 0)[0][0]
        num_modes_ = num_modes if num_modes is not None else eff_dim

        for nm_ in range(0, num_modes_, 4):
            # N = nm_ if nm_ is not None else eff_dim
            N = nm_ + 1

            # slowest decaying mode
            mode_s = vl[:, :N]  # Ne+Ni, 20

            # change of variable modes
            mode_s_prime = vl_prime[:, :N]  # Ne+Ni, 20

            # projecting output of adjacent noise level to the mode
            r_eq_exc = rearrange(r_eq[:, [nl, nl+1]], 'n l k -> (n l) k')  # n*nl, Ne+Ni
            proj_output = mode_s.T @ r_eq_exc.T  # 20, t*nl

            # projecting input of adjacent noise level to the change of variable mode
            ff_ = rearrange(ff[:, [nl, nl+1], :], 'n l k -> (n l) k')  # n*nl, Ne
            proj_input = mode_s_prime[:f.N_e, :].T @ ff_.T  # 20, t*nl
            
            d_o_pipepnm, scale_o_pipepnm, d_norm_o_pipepnm = get_normalized_dist(proj_output, img, delta)
            d_i_pipepnm, scale_i_pipepnm, d_norm_i_pipepnm = get_normalized_dist(proj_input, img, delta)

            d_o_pipe[nm_//4, nl] = d_o_pipepnm
            scale_o_pipe[nm_//4, nl] = scale_o_pipepnm
            d_norm_o_pipe[nm_//4, nl] = d_norm_o_pipepnm
            d_i_pipe[nm_//4, nl] = d_i_pipepnm
            scale_i_pipe[nm_//4, nl] = scale_i_pipepnm
            d_norm_i_pipe[nm_//4, nl] = d_norm_i_pipepnm

    return d_o_pipe, scale_o_pipe, d_norm_o_pipe, d_i_pipe, scale_i_pipe, d_norm_i_pipe


if __name__ == '__main__':
    for pp in patterns:
        # output dir
        out_dir = f'proj_dists_slow_1-200/sample_set{pp}'
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        
        jac_out_dir = 'example_jac_matrix'
        Path(jac_out_dir).mkdir(exist_ok=True, parents=True)

        # load input ----------------------------------------------------------------------------------
        ff = np.load(args.input_path)
        ff = rearrange(ff, 'n l p k h w -> (n l p) (k h w)')  # n_img*nl*np, Ne
        # ------------------------------------------------------------------------------------------------------------------
        # pca of inputs
        pca = PCA(n_components=200)
        ff_reduced = pca.fit_transform(ff)  # t*n_img*nl*np, k

        # calculate all distance
        dist_o = np.zeros((modes_ns[0]//4, 5, 3, num_imgs))
        var_o = np.zeros((modes_ns[0]//4, 5, 3, num_imgs))
        ddc_o = np.zeros((modes_ns[0]//4, 5, 3, num_imgs))
        dist_i = np.zeros((modes_ns[0]//4, 5, 3, num_imgs))
        var_i = np.zeros((modes_ns[0]//4, 5, 3, num_imgs))
        ddc_i = np.zeros((modes_ns[0]//4, 5, 3, num_imgs))
        dist_pre_o = np.zeros((modes_ns[0]//4, 3, num_imgs))
        var_pre_o = np.zeros((modes_ns[0]//4, 3, num_imgs))
        ddc_pre_o = np.zeros((modes_ns[0]//4, 3, num_imgs))
        dist_pre_i = np.zeros((modes_ns[0]//4, 3, num_imgs))
        var_pre_i = np.zeros((modes_ns[0]//4, 3, num_imgs))
        ddc_pre_i = np.zeros((modes_ns[0]//4, 3, num_imgs))
        dist_ff = np.zeros((3, num_imgs))
        var_ff = np.zeros((3, num_imgs))
        ddc_ff = np.zeros((3, num_imgs))

        for n, nm in zip(range(num_imgs), modes_ns):
            # alpha-space ----------------------------------------------------------------------------------
            ff_reduced_ = rearrange(ff_reduced, '(n l p) k -> n l p k', n=num_imgs, l=4, p=10)
            v_ff = [rearrange(ff_reduced_[:, [i, i+1], pp], 'n l k -> k (n l)') for i in [0, 1, 2]]  # pc, n*l
            
            for i, (v_ff_, delta) in enumerate(zip(v_ff, [1, 2, 2])):
                d_ff_pi, scale_ff_pi, d_norm_ff_pi = get_normalized_dist(v_ff_, n, delta)
                # ------------------------------------------------------------------------------------------------------------------
                
                dist_ff[i, n] = d_ff_pi
                var_ff[i, n] = scale_ff_pi
                ddc_ff[i, n] = d_norm_ff_pi

            # default network -------------------------------------------------------------------------------
            (
                d_o_pre_pi, 
                scale_o_pre_pi, 
                d_norm_o_pre_pi, 
                d_i_pre_pi, 
                scale_i_pre_pi, 
                d_norm_i_pre_pi
                ) = analyze_eigenmodes_pre_v3(img=n, pp=pp, num_modes=nm)
            # ------------------------------------------------------------------------------------------------------------------
        
            dist_pre_o[:, :, n] = d_o_pre_pi
            var_pre_o[:, :, n] = scale_o_pre_pi
            ddc_pre_o[:, :, n] = d_norm_o_pre_pi
            dist_pre_i[:, :, n] = d_i_pre_pi
            var_pre_i[:, :, n] = scale_i_pre_pi
            ddc_pre_i[:, :, n] = d_norm_i_pre_pi

            # trained network
            dist_o_pi, var_o_pi, ddc_o_pi = [], [], []
            dist_i_pi, var_i_pi, ddc_i_pi = [], [], []
            for e in [1, 2, 3, 4, 5]:
                # --------------------------------------------------------------------------------------
                (
                    d_o_pipe, 
                    scale_o_pipe, 
                    d_norm_o_pipe, 
                    d_i_pipe, 
                    scale_i_pipe, 
                    d_norm_i_pipe
                    ) = analyze_eigenmodes_trained_v3(epoch=e, img=n, pp=pp, num_modes=nm)
                # ------------------------------------------------------------------------------------------------------------

                dist_o[:, e-1, :, n] = d_o_pipe
                var_o[:, e-1, :, n] = scale_o_pipe
                ddc_o[:, e-1, :, n] = d_norm_o_pipe
                dist_i[:, e-1, :, n] = d_i_pipe
                var_i[:, e-1, :, n] = scale_i_pipe
                ddc_i[:, e-1, :, n] = d_norm_i_pipe

            print(f'[INFO] pattern={pp}; img={n} done.')

        np.save(os.path.join(out_dir, 'dist_o.npy'), dist_o)  # nm, day, nl, img
        np.save(os.path.join(out_dir, 'var_o.npy'), var_o)  # nm, day, nl, img
        np.save(os.path.join(out_dir, 'ddc_o.npy'), ddc_o)  # nm, day, nl, img
        np.save(os.path.join(out_dir, 'dist_pre_o.npy'), dist_pre_o)  # nm, nl, img
        np.save(os.path.join(out_dir, 'var_pre_o.npy'), var_pre_o) # nm, nl, img
        np.save(os.path.join(out_dir, 'ddc_pre_o.npy'), ddc_pre_o) # nm, nl, img
        np.save(os.path.join(out_dir, 'dist_i.npy'), dist_i)  # nm, day, nl, img
        np.save(os.path.join(out_dir, 'var_i.npy'), var_i)  # nm, day, nl, img
        np.save(os.path.join(out_dir, 'ddc_i.npy'), ddc_i)  # nm, day, nl, img
        np.save(os.path.join(out_dir, 'dist_pre_i.npy'), dist_pre_i)  # nm, nl, img
        np.save(os.path.join(out_dir, 'var_pre_i.npy'), var_pre_i) # nm, nl, img
        np.save(os.path.join(out_dir, 'ddc_pre_i.npy'), ddc_pre_i) # nm, nl, img
        np.save(os.path.join(out_dir, 'dist_ff.npy'), dist_ff)  # nl, img
        np.save(os.path.join(out_dir, 'var_ff.npy'), var_ff)  # nl, img
        np.save(os.path.join(out_dir, 'ddc_ff.npy'), ddc_ff)  # nl, img
