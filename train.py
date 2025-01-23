from utils import *
from model import ProtoRNN
import os.path as osp
import numpy as np
import torch
import argparse
from pathlib import Path
from einops import rearrange, repeat

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
    parser.add_argument("--tau_stim", type=int, default=300, help='presentation period for one stimulus')
    parser.add_argument("--delta_t", type=int, default=1, help="step size for the newton method")
    parser.add_argument("--gamma", type=int, default=30, help="feedfoward scaling, -> saliency/attentional signal")
    parser.add_argument("--save_every_stim", type=int, default=1, 
                        help="how much time step to save during the stimulus presentation period. If set to 1, then always save the last")
    # model hyperparam
    parser.add_argument("--wie", type=float, default=30.0, help='fixed all E-I connection strength')
    
    # training param
    parser.add_argument("--epoch", type=int, default=5, help="maximum number of epoch")
    parser.add_argument("--start_epoch", type=int, default=0, help="No. epoch to start training")
    parser.add_argument("--set_wrp", type=parse_boolean, default=False, 
                        help='whether to load previous weight. If this is true, wrp_path must not be empty')
    parser.add_argument("--wrp_path", type=str, default='')
    parser.add_argument("--r_in_path", type=str, required=True)
    parser.add_argument("--theta_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--postfix", type=str, default='', help='name postfix')
    parser.add_argument("--post_only", type=parse_boolean, default=False, help='only save last epoch')
    parser.add_argument("--task", type=str, required=True, 
                        help='effetcs: familiarity effect experiment; assoc: familiarity association experiment')

    return parser.parse_args()


args = parse_args()

# create dir
Path(args.out_path).mkdir(parents=True, exist_ok=True)

# init model
f = ProtoRNN(args.num_kernel, args.num_row, args.num_col, 
             args.tau_stim, args.delta_t, args.save_every_stim, 
             wie=args.wie)

# initial BCM threshold
theta_BCM = np.load(args.theta_path)
f.theta = theta_BCM

# set wrp [Optional]
if args.set_wrp:
    wrp = np.load(args.wrp_path)
    f.wrp = wrp
    print('>> old weight loaded')

# move to gpu
f.to(device)

if __name__ == '__main__':
    # get stimuli for running
    r_in = np.load(args.r_in_path)
    
    if args.task == 'assoc':
        # for noise task
        img_num, nl, npt = r_in.shape[0], r_in.shape[1], r_in.shape[2]
        r_in, _, _ = insertTeachingSig(r_in, p=args.mix_coef)  # n_img, nl*n+nl*n//(p-1)+p, C, H, W
        r_in = torch.tensor(r_in).reshape(-1, 1, f.num_kernel, f.num_row, f.num_col).float().to(device)

    elif args.task == 'effects':
        # for all image task
        img_num = r_in.shape[0]
        r_in = torch.tensor(r_in).reshape(-1, 1, f.num_kernel, f.num_row, f.num_col).float().to(device)

    # set simulation params
    f.gamma = args.gamma

    for e in range(args.start_epoch, args.epoch):
        # large iputs
        # ys_all = []
        # for i in range(img_num):
        #     r_in_sing = rearrange(torch.tensor(r_in[[i]]).float().to(device), 'n s p c h w -> (n s) p c h w')  # n_imgx(nl-1)xnp, 2*n_repeat, C, H, W

        #     # print(f'start epoch{e} ...')
        #     ys, th, above_th = f.train(e, r_in_sing, 0)
        #     ys = ys.squeeze()
        #     ys_all.append(ys)
        
        # ys_all = np.concatenate(ys_all, 0)

        # small inputs
        ys, th, above_th = f.train(e, r_in, 0)
        ys = ys.squeeze()

        if args.task == 'effects':
            # save img exp
            wname = f'weight_epoch{e}_tx{args.gamma}_{args.tau_stim}ms_{img_num}_{args.num_kernel}_{args.num_row}_{args.num_col}_{int(args.wie)}_{args.postfix}.npy'
            tname = f'theta_epoch{e}_tx{args.gamma}_{args.tau_stim}ms_{img_num}_{args.num_kernel}_{args.num_row}_{args.num_col}_{int(args.wie)}_{args.postfix}.npy'

        elif args.task == 'assoc':
            # save noise exp
            wname = f'weight_epoch{e}_tx{args.gamma}_{args.tau_stim}ms_{img_num}_{nl}x{npt}_{args.num_kernel}_{args.num_row}_{args.num_col}_{args.postfix}.npy'
            tname = f'theta_epoch{e}_tx{args.gamma}_{args.tau_stim}ms_{img_num}_{nl}x{npt}_{args.num_kernel}_{args.num_row}_{args.num_col}_{args.postfix}.npy'
            ysname = f'ys_epoch{e}_tx{args.gamma}_{args.tau_stim}ms_{img_num}_{nl}x{npt}_{args.num_kernel}_{args.num_row}_{args.num_col}_{args.postfix}.npy'
        
        itv = 5 if args.post_only else 1
        if (e+1) % itv == 0:
            np.save(osp.join(args.out_path, wname), f.wrp.cpu().numpy())
            np.save(osp.join(args.out_path, tname), f.theta.cpu().numpy())
        