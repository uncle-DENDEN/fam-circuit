from utils import *
from model import ProtoRNN
import os.path as osp
import numpy as np
import torch
import argparse
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False

    return False


def parse_args():
    parser = argparse.ArgumentParser("Training ProtoRNN")
    # image param
    parser.add_argument("--num_kernel", type=int, default=64, help="number of kernels")
    parser.add_argument("--num_row", type=int, default=8, help="number of rows in 2D rectilinear grid")
    parser.add_argument("--num_col", type=int, default=8, help="number of columns in 2D rectilinear grid")
    parser.add_argument("--mix_coef", type=int, default=2, help="coefficient for mixing teaching signal and noise")

    # simulation param
    parser.add_argument("--tau_stim", type=int, default=150, help='presentation period for one stimulus')
    parser.add_argument("--delta_t", type=int, default=1, help="step size for the newton method")
    parser.add_argument("--gamma", type=int, default=1, help="feedfoward scaling, -> saliency/attentional signal")
    parser.add_argument("--save_every_stim", type=int, default=150, 
                        help="how much time step to save during the stimulus presentation period. If set to 1, then always save the last")
    # model hyperparam
    parser.add_argument("--wie", type=float, default=30.0, help='fixed all E-I connection strength')
    
    # testing param
    parser.add_argument("--set_wrp", type=parse_boolean, default=True)
    parser.add_argument("--wrp_path", type=str, default='')
    parser.add_argument("--r_in_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--postfix", type=str, default='', help='name postfix')
    parser.add_argument("--task", type=str, required=True, help='effetcs: familiarity effect experiment; assoc: familiarity association experiment')
    parser.add_argument("--get_BCM_threshold", type=parse_boolean, default=False, help='compute BCM threshold after testing')

    return parser.parse_args()

args = parse_args()

# create dir
Path(args.out_path).mkdir(parents=True, exist_ok=True)

# init model
f = ProtoRNN(args.num_kernel, args.num_row, args.num_col, args.tau_stim, 
             args.delta_t, args.save_every_stim, 
             wie=args.wie)

# set wrp and theta [Optional]
if args.set_wrp:
    wrp = np.load(args.wrp_path)
    f.wrp = wrp

# move to gpu
f.to(device)

if __name__ == '__main__':
    # get stimuli for running
    r_in = np.load(args.r_in_path)
    
    if args.task == 'assoc':
        img_num, nl, npt = r_in.shape[0], r_in.shape[1], r_in.shape[2]
        r_in, t_ind, n_ind = insertTeachingSig(r_in, p=args.mix_coef)
        r_in = torch.tensor(r_in).reshape(-1, 1, f.num_kernel, f.num_row, f.num_col).float().to(device)
    
    elif args.task == 'effects':
        img_num = r_in.shape[0]
        r_in = torch.tensor(r_in).reshape(-1, 1, f.num_kernel, f.num_row, f.num_col).float().to(device)

    # set hyperparams
    f.wie_ = args.wie

    # run training
    # for i in trange(n_trials, desc='trials...', position=0):
    ys_test, _ = f.test(r_in, 0)

    # save
    if args.task == 'assoc':
        if args.set_wrp:
            train_info = args.wrp_path.split('/')[-1].split('_')[1:3]
            ti = f'{train_info[0]}_{train_info[1]}'
        else:
            ti = 'pre'
        wname = f'r_{ti}_{args.tau_stim}ms_{img_num}_{nl}x{npt}_{args.save_every_stim}_{args.num_kernel}_{args.num_row}_{args.num_col}_{args.postfix}.npy'

    elif args.task == 'effects':
        if args.set_wrp:
            train_info = args.wrp_path.split('/')[-1].split('_')[1:3]
            ti = f'{train_info[0]}_{train_info[1]}'
        else:
            ti = 'pre'
        wname = f'r_{ti}_{args.tau_stim}ms_{img_num}_{args.save_every_stim}_{args.num_kernel}_{args.num_row}_{args.num_col}_{int(args.wie)}_{args.postfix}.npy'
    
    np.save(osp.join(args.out_path, wname), ys_test)

    if args.get_BCM_threshold:
        if args.task == 'assoc':
            ys_test = seqUnmix(ys_test, t_ind, n_ind, n_imgs=5, npa=10)
        thresh = get_BCM_thresh(f, ys_test)

        r_in_path = Path(args.r_in_path)
        input_dir = str(r_in_path.parent)
        r_in_file = r_in_path.stem
        if args.task == 'effects':
            thresh_name = '_'.join(['BCM', 'theta'] + r_in_file.split('_')[2:]) + f'_ys_{int(args.wie)}_{args.postfix}' + '.npy'
        elif args.task == 'assoc':
            thresh_name = '_'.join(['BCM', 'theta'] + r_in_file.split('_')[2:]) + f'_ys_{args.postfix}' + '.npy'
        np.save(osp.join(input_dir, thresh_name), thresh)
