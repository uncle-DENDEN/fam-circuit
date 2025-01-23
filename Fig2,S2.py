from utils import *
from model import ProtoRNN
from scipy.stats import pearsonr, ttest_1samp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import seaborn as sns
import numpy as np
import torch
import argparse
import re

## params
def parse_args():
    parser = argparse.ArgumentParser("Fig2, S2")
    parser.add_argument("--pre_path", type=str, required=True)
    parser.add_argument("--post_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--wies", type=float, nargs="+", default=[10, 15, 20, 25, 30, 35, 40, 45, 50])

    return parser.parse_args()

args = parse_args()

## figure parameter
SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

## Model
f = ProtoRNN(64, 8, 8, 150, 1, 75, wie=30)
f.to(torch.device("cpu"))

## assoc exp input info
if args.task == 'assoc':
    r_in = np.load('input_famassoc/r_in_cifar_all_noise_abs_5_4_10_64_8_8.npy')
    r_in, t_ind, n_ind = insertTeachingSig(r_in, p=2)

## plot functions
def sparseness(tuning):
    """
    Calculate the tuning sparsity according to 

    :param tuning: neural tuning of shape (n_stimulus, n_neuron)
    :return: sparsity of tuning for each neuron
    """
    n = tuning.shape[0]
    return (1 - ((tuning.sum(0)/n)**2/((tuning**2/n).sum(0) + 1e-8)))/(1 - (1/n))


def replace_wie(filename, wie):
    pattern = r'8_8_(\d+)'
    return re.sub(pattern, f'8_8_{wie}', filename)


def Fig_2B_S3A(pre_path, post_path, task='effects'):

    # load input
    ys_before = np.load(pre_path)
    ys_after = np.load(post_path)

    if task == 'assoc':
        ys_before = seqUnmix(ys_before, t_ind, n_ind, n_imgs=5, npa=10)
        ys_after = seqUnmix(ys_after, t_ind, n_ind, n_imgs=5, npa=10)
    
    ys_before_exc = f.r_numpy(ys_before[:, :, :f.N_e])
    ys_after_exc = f.r_numpy(ys_after[:, :, :f.N_e])
    # averaged plot for all neurons
    ys_mean_before = ys_before_exc.mean((0, 2))
    ys_mean_after = ys_after_exc.mean((0, 2))

    psth_after = np.insert(ys_mean_after, 0, 0)
    psth_after = psth_after / psth_after.max()
    psth_before = np.insert(ys_mean_before, 0, 0)
    psth_before = psth_before / psth_before.max()

    figsize=(4, 3.5) if task == 'assoc' else (7, 3.5)
    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)
    x = np.insert(np.arange(0, 220, 2), 0, -10)
    axs.plot(x, psth_before[:111], '-', label='before', color='black')
    axs.plot(x, psth_after[:111], "--", label='after', color='black')
    axs.fill_between(x, psth_before[:111], psth_after[:111], color='yellow', alpha=.5)
    axs.axvline(x=0, linestyle='dashed', linewidth=2, color='black')
    axs.legend(frameon=False)
    axs.set_xlabel('Time (msec)')
    axs.set_ylabel('average normalized response')

    axs.spines['bottom'].set_position(('data', 0))
    axs.spines['left'].set_position(('data', -10))

    # Make top and right spines invisible
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    axs.set_xticks([0, 50, 100, 200], [0, 50, 100, 200])
    axs.set_yticks([0.2, 0.6, 1.0], [0.2, 0.6, 1.0])

    # save dir and name
    save_dir = 'figs/fig2' if task == 'effects' else 'figs/figS3'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_name = 'fig2B.png' if task == 'effects' else 'figS3A.png'
    fig.savefig(os.path.join(save_dir, save_name))


def Fig2CDE_S3BC(pre_path, post_path, task='effects'):
    
    # load input
    tm_before = np.load(pre_path)
    tm_after = np.load(post_path)

    if task == 'assoc':
        tm_before = seqUnmix(tm_before, t_ind, n_ind, n_imgs=5, npa=10)[:, -2:].mean(1)
        tm_after = seqUnmix(tm_after, t_ind, n_ind, n_imgs=5, npa=10)[:, -2:].mean(1)
    elif task == 'effects':
        tm_before = tm_before[:, -2:].mean(1)
        tm_after = tm_after[:, -2:].mean(1)
    
    tm_before = f.r_numpy(tm_before[:, :f.N_e])
    tm_after = f.r_numpy(tm_after[:, :f.N_e])

    # save dor
    save_dir = 'figs/fig2' if task == 'effects' else 'figs/figS3'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # sparsity vs max tuning
    max_before = tm_before.max(0)
    max_after = tm_after.max(0)
    max_diff = (max_after - max_before) / (max_after + max_before + 1e-8)

    _, p = ttest_1samp(max_diff, 0, alternative='greater')

    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(4, 2.2))
    sns.histplot(max_diff, bins=100, kde=True, ax=axs)
    axs.set_xlabel(r'$\Delta$ response peak')
    _, ymax = axs.get_ylim()
    axs.plot(max_diff.mean(), ymax, marker='v', markersize=7, 
            markerfacecolor='black', markeredgecolor='black')
    axs.text(np.mean(max_diff) + 0.1, ymax, "***" if p < 0.05 else 'ns', 
             fontsize=15, verticalalignment='top')
    axs.set_ylabel('')
    axs.set_yticks([], [])

    axs.spines['left'].set_position(('data', 0))
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)

    save_name = 'fig2D.png' if task == 'effects' else 'figS3B.png'
    fig.savefig(os.path.join(save_dir, save_name))

    # Vinji-Gallant sparseness
    sp_before = sparseness(tm_before)
    sp_after = sparseness(tm_after)
    sp_diff = (sp_after - sp_before) / (sp_after + sp_before)

    stat, p = ttest_1samp(sp_diff, 0, alternative='greater')

    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(4, 2.2))
    sns.histplot(sp_diff, bins=100, kde=True, ax=axs)
    _, ymax = axs.get_ylim()
    axs.plot(np.mean(sp_diff), ymax, marker='v', markersize=7, 
            markerfacecolor='black', markeredgecolor='black')
    axs.text(np.mean(sp_diff) + 0.01, ymax, "***" if p < 0.05 else 'ns', 
             fontsize=15, verticalalignment='top')
    axs.set_ylabel('')
    axs.set_yticks([], [])
    axs.set_xlabel(r'$\Delta$ Sparseness index')

    axs.spines['left'].set_position(('data', 0))
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)

    save_name = 'fig2E.png' if task == 'effects' else 'figS3C.png'
    fig.savefig(os.path.join(save_dir, save_name))

    # tuning of example neuron with large sparsity change
    if task == 'effects':
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(4.2, 4))

        # idx = np.argwhere((sp_diff > 0) & (sp_diff < 0.2))
        idx = np.argwhere(sp_diff > 0.02)
        i = 200
        tm_before_ = np.sort(tm_before[:, idx], axis=0)[::-1]
        tm_after_ = np.sort(tm_after[:, idx], axis=0)[::-1]
        ax.plot(tm_before_[:, i], alpha=.8, label='before', linewidth=2)
        ax.plot(tm_after_[:, i], alpha=.8, label='after', linewidth=2)
        ax.set_xlim(-10, 300)
        # ax.set_ylim(-0.05, 1.5)
        ax.legend(frameon=False)
        ax.set_xlabel('sorted stimulus index')
        ax.set_ylabel('firing rate')

        ax.spines['bottom'].set_position(('data', -0.05))
        ax.spines['left'].set_position(('data', -10))

        # Make top and right spines invisible
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xticks([0, 100, 200, 300], [0, 100, 200, 300])
        # ax.set_yticks([0, 0.4, 0.8, 1.2], [0, 0.4, 0.8, 1.2])

        save_name = 'fig2C.png'
        fig.savefig(os.path.join(save_dir, save_name))

    
def FigS2(pre_path, post_path, wies):

    ## save dir
    save_dir = 'figs/figS2'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    ## suppresion index
    # load data
    ys_before_all = []
    ys_after_all = []
    for wie in wies:
        ys_before = np.load(replace_wie(pre_path, wie))
        ys_after = np.load(replace_wie(post_path, wie))
        ys_before_exc = f.r_numpy(ys_before[:, :, :f.N_e])
        ys_after_exc = f.r_numpy(ys_after[:, :, :f.N_e])
        
        del ys_before, ys_after
        
        ys_mean_before = ys_before_exc.mean((0, 2))  # 250
        ys_mean_after = ys_after_exc.mean((0, 2))  # 250

        del ys_before_exc, ys_after_exc

        ys_after_norm = ys_mean_after / ys_mean_after.max()
        ys_before_norm = ys_mean_before / ys_mean_before.max()
        
        ys_before_all.append(ys_before_norm)
        ys_after_all.append(ys_after_norm)

        del ys_before_norm, ys_after_norm

    ys_before_all = np.stack(ys_before_all)  # 8, 25
    ys_after_all = np.stack(ys_after_all)  # 8, 25 

    sup_ind = (ys_after_all - ys_before_all) / (ys_after_all + ys_before_all)
    sup_ind = sup_ind[:, -5:].mean(-1)  # 9

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(4, 3.3))
    ax.plot(sup_ind, '-o', color='k')
    ax.set_xticks(np.arange(len(wies)), wies)
    ax.set_xlabel('E-I connection strength')
    ax.set_ylabel('suppression index')
    ax.axvline(x=1, linestyle='--', linewidth=2, color='k')
    ax.set_xlim(-0.2, 8.2)

    save_name = 'figS2A.png'
    fig.savefig(os.path.join(save_dir, save_name))

    ## sparsity index
    # load data
    tm_before_all = []
    tm_after_all = []
    for wie in wies:
        tm_before = np.load(replace_wie(pre_path, wie))
        tm_before = tm_before[:, -2:].mean(1)
        tm_after = np.load(replace_wie(post_path, wie))
        tm_after = tm_after[:, -2:].mean(1)

        tm_before = f.r_numpy(tm_before[:, :f.N_e])
        tm_after = f.r_numpy(tm_after[:, :f.N_e])

        tm_before_all.append(tm_before)
        tm_after_all.append(tm_after)

    tm_before_all = np.stack(tm_before_all)  # 8, 500, Ne
    tm_after_all = np.stack(tm_after_all)  # 8, 500, Ne

    # sparsity index
    sp_diff_all = []
    p_all = []
    for i in range(9):
        sp_before = sparseness(tm_before_all[i])
        sp_after = sparseness(tm_after_all[i])
        sp_diff = (sp_after - sp_before) / (sp_after + sp_before + 1e-8)
        sp_diff_all.append(sp_diff.mean())

        stat, p = ttest_1samp(sp_diff, 0, alternative='greater')
        p_all.append(p)
    
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(4, 3.3))
    ax.plot(sp_diff_all, '-o', color='k')
    for i, (y, p) in enumerate(zip(sp_diff_all, p_all)):
        lab = "*" if (p < 0.05) & (not np.isnan(p)) else "ns"
        ax.text(i + 0.1, y, lab, fontsize=15, 
                horizontalalignment='left',
                verticalalignment='center')
    
    ax.axvline(x=1, linestyle='--', linewidth=2, color='k')
    ax.set_xticks(np.arange(len(wies)), wies)
    ax.set_xlabel('E-I connection strength')
    ax.set_ylabel(r'mean $\Delta$ sparseness index')
    ax.set_xlim(-0.2, 8.2)
    # ax.set_ylim(0, 0.0046)
    
    fig = plt.gcf()
    save_name = 'figS2B'
    fig.savefig(os.path.join(save_dir, save_name))


if __name__ == '__main__':
    
    Fig_2B_S3A(args.pre_path, args.post_path, args.task)
    Fig2CDE_S3BC(args.pre_path, args.post_path, args.task)
    FigS2(args.pre_path, args.post_path, args.wies)
    print('== ploting finished ==')
    