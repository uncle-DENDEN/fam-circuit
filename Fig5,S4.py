from utils import *
from model import ProtoRNN

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from numpy.linalg import norm
from scipy.stats import sem, pearsonr
import scipy.stats as stats
import scipy.linalg as linalg
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
import argparse
import joblib
import re

import torch
import torchvision.datasets as ds
import torchvision.transforms.functional as ttf
from einops import rearrange, repeat

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

## model
f = ProtoRNN(64, 8, 8, 150, 1, 75, wie=30)
f.to(torch.device("cpu"))


def replace_epoch(path, epoch):
    epoch_pos = path.find('epoch')
    end_pos = epoch_pos + 5 
    while end_pos < len(path) and path[end_pos].isdigit():
        end_pos += 1
    
    if epoch == 'pre':
        new_path = path[:epoch_pos] + str(epoch) + path[end_pos:]
    else:
        new_path = path[:epoch_pos + 5] + str(epoch) + path[end_pos:]

    return new_path


def replace_sample(path, p):
    return re.sub(r'sample_set\d+', f'sample_set{p}', path)


class plot_mode:
    def __init__(
            self, 
            input_path, 
            pre_response_path, 
            post_response_path, 
            jac_path,
            proj_dist_path
        ) -> None:

        self.postfix = {0: 'clear', 1: '10%', 2: '30%', 3: '50%'}
        self.num_imgs = 5
        
        r_in = np.load(input_path)
        _, self.t_ind, self.n_ind = insertTeachingSig(r_in, p=2)

        self.n_pc = 200
        self.sample_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        self.pre_response_path = pre_response_path
        self.post_response_path = post_response_path
        self.jac_path = jac_path
        self.proj_dist_path = proj_dist_path

        # output_path
        self.fig5_path = 'figs/fig5'
        Path(self.fig5_path).mkdir(parents=True, exist_ok=True)
        self.figS4_path = 'figs/figS4'
        Path(self.figS4_path).mkdir(parents=True, exist_ok=True)

    def get_eigspec(self, epoch, num_modes=8192):
        eigd = joblib.load(replace_epoch(self.jac_path, epoch))
       
        eigspec = eigd['val']
        vl = eigd['vl']
        vl_prime = eigd['vl_prime']

        mode_s = vl[:, :num_modes]  # Ne+Ni, 20
        mode_s_prime = vl_prime[:, :num_modes]  # Ne+Ni, 20
        
        return eigspec, mode_s, mode_s_prime
    
    def get_normalized_dist(self, fp):
        day, nl, img, _ = fp.shape
        ds = [1, 2, 2]

        euc_dist = np.zeros((day, nl-1, img))
        varr = np.zeros((day, nl-1, img))
        ddc = np.zeros((day, nl-1, img))

        for n in range(img):
            for i in range(nl-1):
                fp_l = fp[:, i, n]
                fp_lp1 = fp[:, i+1, n]

                d = ds[i]
                dist = norm((fp_l - fp_lp1) / d, axis=-1)
                euc_dist[:, i, n] = dist

                fp_l_alli = fp[:, i]
                fp_l_ref = np.expand_dims(fp_l, 1)
                cov1 = (fp_l_alli - fp_l_ref).transpose(0, 2, 1) @ (fp_l_alli - fp_l_ref) / (img - 1)
                var_ = np.sqrt(np.trace(cov1, axis1=-2, axis2=-1)) 
                varr[:, i, n] = var_

                dist_norm = dist / var_
                ddc[:, i, n] = dist_norm

        return euc_dist, varr, ddc
    
    def get_response_dist(self):
        ys_pre = np.load(self.pre_response_path)[..., :f.N_e] 
        ys_pre = seqUnmix(ys_pre, self.t_ind, self.n_ind, n_imgs=self.num_imgs, npa=10) 
        ys_pre = f.r_numpy(ys_pre)[:, -2:].mean(-2)

        ys_all = [ys_pre]
        del ys_pre
        for i in range(5):
            ys_e = np.load(replace_epoch(self.post_response_path, i))[..., :f.N_e]
            ys_e = seqUnmix(ys_e, self.t_ind, self.n_ind, n_imgs=self.num_imgs, npa=10)
            ys_e = f.r_numpy(ys_e)[:, -2:].mean(-2)
            ys_all.append(ys_e)
            del ys_e

        ys_all_exc = np.stack(ys_all)

        fp_inp = ys_all_exc.reshape(-1, f.N_e)  # (day*n_img*nl*np, f.N_e)
        pca = PCA(n_components=self.n_pc)
        fp = pca.fit_transform(fp_inp)

        # plt.plot(np.cumsum(pca.explained_variance_ratio_))
        # plt.axvline(x=15, color='black', linestyle='dashed')

        fp = fp[..., :self.n_pc]

        fp = rearrange(fp, '(d n l p) k -> d l n p k', n=self.num_imgs, l=4, p=10)

        euc_dist_a, varr_a, ddc_a = [], [], []
        for p in self.sample_list:
            fp_ = fp[:, :, :, p]
            euc_dist, varr, ddc = self.get_normalized_dist(fp_)
            euc_dist_a.append(euc_dist)
            varr_a.append(varr)
            ddc_a.append(ddc)

        euc_dist = np.stack(euc_dist_a).mean(0)
        varr = np.stack(varr_a).mean(0)
        ddc = np.stack(ddc_a).mean(0)

        return euc_dist, varr, ddc

    def get_proj_dist(self):
        dist_o_a = []
        var_o_a = []
        ddc_o_a = []
        dist_pre_o_a = []
        var_pre_o_a = []
        ddc_pre_o_a = []
        dist_i_a = []
        var_i_a = []
        ddc_i_a = []
        dist_pre_i_a = []
        var_pre_i_a = []
        ddc_pre_i_a = []
        dist_ff_a = []
        var_ff_a = []
        ddc_ff_a = []

        for p in self.sample_list:
            out_dir = replace_sample(self.proj_dist_path, p)
            
            dist_o = np.load(os.path.join(out_dir, 'dist_o.npy')) 
            var_o = np.load(os.path.join(out_dir, 'var_o.npy')) 
            ddc_o = np.load(os.path.join(out_dir, 'ddc_o.npy')) 
            dist_pre_o = np.load(os.path.join(out_dir, 'dist_pre_o.npy'))
            var_pre_o = np.load(os.path.join(out_dir, 'var_pre_o.npy'))
            ddc_pre_o = np.load(os.path.join(out_dir, 'ddc_pre_o.npy')) 
            dist_i = np.load(os.path.join(out_dir, 'dist_i.npy')) 
            var_i = np.load(os.path.join(out_dir, 'var_i.npy')) 
            ddc_i = np.load(os.path.join(out_dir, 'ddc_i.npy')) 
            dist_pre_i = np.load(os.path.join(out_dir, 'dist_pre_i.npy'))
            var_pre_i = np.load(os.path.join(out_dir, 'var_pre_i.npy'))
            ddc_pre_i = np.load(os.path.join(out_dir, 'ddc_pre_i.npy'))
            dist_ff = np.load(os.path.join(out_dir, 'dist_ff.npy')) 
            var_ff = np.load(os.path.join(out_dir, 'var_ff.npy'))
            ddc_ff = np.load(os.path.join(out_dir, 'ddc_ff.npy'))

            # Append each array to its corresponding list
            dist_o_a.append(dist_o)
            var_o_a.append(var_o)
            ddc_o_a.append(ddc_o)
            dist_pre_o_a.append(dist_pre_o)
            var_pre_o_a.append(var_pre_o)
            ddc_pre_o_a.append(ddc_pre_o)
            dist_i_a.append(dist_i)
            var_i_a.append(var_i)
            ddc_i_a.append(ddc_i)
            dist_pre_i_a.append(dist_pre_i)
            var_pre_i_a.append(var_pre_i)
            ddc_pre_i_a.append(ddc_pre_i)
            dist_ff_a.append(dist_ff)
            var_ff_a.append(var_ff)
            ddc_ff_a.append(ddc_ff)

        # Concatenate and average
        dist_o = np.mean(np.stack(dist_o_a, axis=0), axis=0)
        var_o = np.mean(np.stack(var_o_a, axis=0), axis=0)
        ddc_o = np.mean(np.stack(ddc_o_a, axis=0), axis=0)
        dist_pre_o = np.mean(np.stack(dist_pre_o_a, axis=0), axis=0)
        var_pre_o = np.mean(np.stack(var_pre_o_a, axis=0), axis=0)
        ddc_pre_o = np.mean(np.stack(ddc_pre_o_a, axis=0), axis=0)
        dist_i = np.mean(np.stack(dist_i_a, axis=0), axis=0)
        var_i = np.mean(np.stack(var_i_a, axis=0), axis=0)
        ddc_i = np.mean(np.stack(ddc_i_a, axis=0), axis=0)
        dist_pre_i = np.mean(np.stack(dist_pre_i_a, axis=0), axis=0)
        var_pre_i = np.mean(np.stack(var_pre_i_a, axis=0), axis=0)
        ddc_pre_i = np.mean(np.stack(ddc_pre_i_a, axis=0), axis=0)
        dist_ff = np.mean(np.stack(dist_ff_a, axis=0), axis=0)
        var_ff = np.mean(np.stack(var_ff_a, axis=0), axis=0)
        ddc_ff = np.mean(np.stack(ddc_ff_a, axis=0), axis=0)

        return (
            dist_o, var_o, ddc_o, 
            dist_pre_o, var_pre_o, ddc_pre_o, 
            dist_i, var_i, ddc_i, 
            dist_pre_i, var_pre_i, ddc_pre_i,
            dist_ff, var_ff, ddc_ff
        )
    
    def Fig5AB(self):
        # 5A
        eigs, _, vl_prime = self.get_eigspec(epoch=4)
        timescale = -1 / eigs

        vl_prime_norm = np.linalg.norm(vl_prime, axis=0)
        eff_dim = np.where(vl_prime_norm == 0)[0][0]
        g_dim = np.where(vl_prime_norm == 0)[0][-1] + 1

        fig = plt.figure(figsize=(10, 3))
        gs = fig.add_gridspec(1, 10, wspace=0.05, hspace=0.05)
        ax1 = fig.add_subplot(gs[0, 0:3])
        ax2 = fig.add_subplot(gs[0, 3])
        ax3 = fig.add_subplot(gs[0, 4:7])
        ax4 = fig.add_subplot(gs[0, 7:10])

        ax1.plot(np.arange(0, eff_dim), timescale[:eff_dim], 'bo', markerfacecolor='none')
        ax1.set_ylim(-8, 39)
        ax1.set_xticks([0], [0])
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)

        ax2.plot(np.arange(eff_dim, g_dim), timescale[eff_dim: g_dim], '--', linewidth=1, color='tab:red')
        ax2.set_ylim(-8, 39)
        ax2.set_xticks([eff_dim], [eff_dim])
        ax2.set_yticks([], [])
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax3.plot(np.arange(g_dim, 4097), timescale[g_dim: 4097], 'bo', markerfacecolor='none')
        ax3.set_ylim(-8, 39)
        ax3.set_xticks([g_dim], [g_dim])
        ax3.set_yticks([], [])
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)

        ax4.plot(np.arange(4097, 8192), timescale[4097:], 'bo', markerfacecolor='none')
        ax4.set_ylim(-8, 39)
        ax4.set_xticks([4097, 8191], [4097, 8191])
        ax4.set_yticks([], [])
        ax4.spines['right'].set_visible(False)
        ax4.spines['top'].set_visible(False)

        ax1.set_ylabel(r'time constant $\tau$')
        plt.savefig(os.path.join(self.fig5_path, 'fig5A.png'))

        # 5B
        group1s, group3s, group4s = [], [], []
        for i in range(self.num_imgs):
            for l in [0, 1, 2]:
                eigs, _, vl_prime = self.get_eigspec(epoch='pre')
                
                vl_prime_norm = np.linalg.norm(vl_prime, axis=0)
                eff_dim = np.where(vl_prime_norm == 0)[0][0]
                g_dim = np.where(vl_prime_norm == 0)[0][-1] + 1
                group1 = vl_prime[:, :eff_dim]
                group3 = vl_prime[:, g_dim:f.N_e]
                group4 = vl_prime[:, f.N_e:]
                
                group1s.append(group1)
                group3s.append(group3)
                group4s.append(group4)

                for e in range(5):
                    eigs, _, vl_prime = self.get_eigspec(epoch=e)

                    vl_prime_norm = np.linalg.norm(vl_prime, axis=0)
                    eff_dim = np.where(vl_prime_norm == 0)[0][0]
                    g_dim = np.where(vl_prime_norm == 0)[0][-1] + 1
                    group1 = vl_prime[:, :eff_dim]
                    group3 = vl_prime[:, g_dim:f.N_e]
                    group4 = vl_prime[:, f.N_e:]
                    
                    group1s.append(group1)
                    group3s.append(group3)
                    group4s.append(group4)

        group1 = np.concatenate(group1s, 1)
        group3 = np.concatenate(group3s, 1)
        group4 = np.concatenate(group4s, 1)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True, figsize=(9, 1))
        ax1.plot(np.arange(f.N_e), group1.mean(1)[:f.N_e], '-', color='b')
        ax1.plot(np.arange(f.N_e, f.N_e+f.N_i), group1.mean(1)[f.N_e:], '-', color='grey')
        ax1.set_xticks([], [])
        ax1.spines.right.set_visible(False)
        ax1.spines.top.set_visible(False)
        ax1.spines.bottom.set_visible(False)
        ax1.spines.left.set_visible(False)
        ax1.axvline(x=f.N_e, linestyle='dashed', linewidth=2, color='k')

        ax2.plot(np.arange(f.N_e), group3.mean(1)[:f.N_e], '-', color='b')
        ax2.plot(np.arange(f.N_e, f.N_e+f.N_i), group3.mean(1)[f.N_e:], '-', color='grey')
        ax2.set_xticks([], [])
        ax2.spines.right.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax2.spines.bottom.set_visible(False)
        ax2.spines.left.set_visible(False)
        ax2.axvline(x=f.N_e, linestyle='dashed', linewidth=2, color='k')

        ax3.plot(np.arange(f.N_e), group4.mean(1)[:f.N_e], '-', color='b')
        ax3.plot(np.arange(f.N_e, f.N_e+f.N_i), group4.mean(1)[f.N_e:], '-', color='grey')
        ax3.set_xticks([], [])
        ax3.spines.right.set_visible(False)
        ax3.spines.top.set_visible(False)
        ax3.spines.bottom.set_visible(False)
        ax3.spines.left.set_visible(False)
        ax3.axvline(x=f.N_e, linestyle='dashed', linewidth=2, color='k')

        plt.savefig(os.path.join(self.fig5_path, 'fig5B.png'))

    def Fig5C_S4BC(self):
        (
            dist_o, var_o, ddc_o, 
            dist_pre_o, var_pre_o, ddc_pre_o, 
            dist_i, var_i, ddc_i, 
            dist_pre_i, var_pre_i, ddc_pre_i,
            dist_ff, var_ff, ddc_ff
        ) = self.get_proj_dist()

        euc_dist, varr, ddc = self.get_response_dist()
        
        # comparison between four measures -- ddc
        nnd = 49
        data = [ddc_ff, ddc_i[(nnd-1)//4, -1], ddc[-1], ddc_o[(nnd-1)//4, -1]]

        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(5.4, 4))
        x = np.arange(3)
        ax.set_xticks(np.arange(3), ['0%->10%', '10%->30%', '30%->50%'])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6], [0.0, 0.2, 0.4, 0.6])
        labels = [r'$\alpha$', r'$\tilde{L}^T \alpha$', r'$r^e$', r'$L^T r$']
        fmts = ['--o', '-^', '--o', '-^']
        dodge = [-0.075, -0.025, 0.025, 0.075]
        for i, mat in enumerate(data): 
            x_ = x + dodge[i]
            mat_mean = mat.mean(-1)
            mat_sem = sem(mat, -1)
            label = labels[i]
            fmt = fmts[i]
            col ="#0072BD" if (i == 0 or i == 1) else "firebrick"
            ax.errorbar(x_, mat_mean, yerr=mat_sem, fmt=fmt, label=label,
                        color=col, elinewidth=2.6, capsize=5, markersize=8)
        ax.text(0.2, 0.5, r'$\alpha$', fontsize=15)
        ax.text(0.2, 0.29, r'$r^e$', fontsize=15)
        ax.text(0.17, 0.18, r'$\tilde{L}^T \alpha$', fontsize=15)
        ax.text(0.22, 0.06, r'$L^T r$', fontsize=15)
        ax.set_ylabel(r'$\mathcal{D}_{N/C}$')
        ax.set_ylim(0, 0.61)
        
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        fig.savefig(os.path.join(self.fig5_path, 'fig5C.png'))

        # diff ~ subspace dim
        data = [ddc_ff, ddc_i[:, -1], ddc[-1], ddc_o[:, -1]] 

        dodge = 0.3
        fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 4))
        x_plot = np.arange(0, 200, 4)+1
        fmt = ['--', '-', ':']
        labels = ['0%-10%', '10%-30%', '30%-50%']
        # cols = ("#7E2F8E", "#0072BD", "#77AC30")
        for nl in range(3):
            di = (data[1][:, nl] - np.expand_dims(data[0][nl], 0)) / np.expand_dims(data[0][nl], 0)
            do = (data[3][:, nl] - np.expand_dims(data[2][nl], 0)) / np.expand_dims(data[2][nl], 0)
            di_mean = di.mean(-1)
            di_sem = sem(di, -1)
            do_mean = do.mean(-1)
            do_sem = sem(do, -1)
            
            # plot
            ax1.plot(x_plot, di_mean, fmt[nl], label=labels[nl], color=plt.cm.plasma((nl+1)/4))
            ax1.fill_between(x_plot, di_mean+di_sem, di_mean-di_sem, alpha=.3, color=plt.cm.plasma((nl+1)/4))
            ax1.scatter([49], [di_mean[12]], marker='o', edgecolor='white', facecolor='k', zorder=2)
            ax2.plot(x_plot, do_mean, fmt[nl], label=labels[nl], color=plt.cm.plasma((nl+1)/4))
            ax2.fill_between(x_plot, do_mean+do_sem, do_mean-do_sem, alpha=.3, color=plt.cm.plasma((nl+1)/4))
            ax2.scatter([49], [do_mean[12]], marker='o', edgecolor='white', facecolor='k', zorder=2)
            ax2.legend(frameon=False, loc='upper right')
            ax1.set_ylabel(r'$\mathcal{D}_{N/C}$ relative difference')
            ax1.set_title(r'$\tilde{L}^T \alpha$ vs $\alpha$')
            ax1.set_xlabel('subspace dimension')
            ax1.set_ylim(-0.83, -0.05)
            ax1.set_yticks([-0.7, -0.4, -0.1], [-0.7, -0.4, -0.1])
            ax2.set_ylabel(r'$\mathcal{D}_{N/C}$ relative difference')
            ax2.set_title(r'$L^T r$ vs $r^e$')
            ax2.set_xlabel('subspace dimension')
            ax2.set_ylim(-0.79, -0.05)
            ax2.set_yticks([-0.7, -0.4, -0.1], [-0.7, -0.4, -0.1])

            ax1.spines['left'].set_position(('data', 0))
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)

            ax2.spines['left'].set_position(('data', 0))
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
        
        fig.savefig(os.path.join(self.figS4_path, 'figS4BC.png'))

    def Fig5DEF_S4DE(self):
        (
            dist_o, var_o, ddc_o, 
            dist_pre_o, var_pre_o, ddc_pre_o, 
            dist_i, var_i, ddc_i, 
            dist_pre_i, var_pre_i, ddc_pre_i,
            dist_ff, var_ff, ddc_ff
        ) = self.get_proj_dist()

        euc_dist, varr, ddc = self.get_response_dist()

        nnd = 49
        ddc_all = [[ddc_o[(nnd-1)//4], ddc_pre_o[(nnd-1)//4]], [ddc_i[(nnd-1)//4], ddc_pre_i[(nnd-1)//4]], [ddc[1:], ddc[0]]]
        titles = [r'$L^T r$', r'$\tilde{L}^T \alpha$', r'$r^e$']
        cols = ("#7E2F8E", "#0072BD", "#77AC30")

        dodge = 0.3
        fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(12, 3.4))
        locs = ['upper left', 'upper right', 'lower right']
        # cols = ("#7E2F8E", "#0072BD", "#77AC30")
        for i, (ax, ddc_pair) in enumerate(zip(axs, ddc_all)):
            post, pre = ddc_pair 
            post_mean = post.mean(-1)[:, :] 
            post_sem = sem(post, -1)[:, :]
            pre_mean = pre.mean(-1)[:]
            pre_sem = sem(pre, -1)[:] 

            # # nl as x axis
            # x = np.arange(3)
            ax.plot([], [], 'x', color='black', label='pre')
            for m in range(3):
                ax.errorbar(m - dodge, pre_mean[m], yerr=pre_sem[m], fmt='x', 
                            color=plt.cm.plasma((m+1)/4), elinewidth=1.6, capsize=2, )
            ax.plot([], [], 'o', color='black', label='epoch1')
            for m in range(3):
                ax.errorbar(m, post_mean[0, m], yerr=post_sem[0, m], fmt='o', 
                            color=plt.cm.plasma((m+1)/4), elinewidth=1.6, capsize=2, )
            ax.plot([], [], '^', color='black', label='epoch5')
            for m in range(3):
                ax.errorbar(m + dodge, post_mean[-1, m], yerr=post_sem[-1, m], fmt='^', 
                            color=plt.cm.plasma((m+1)/4), elinewidth=1.6, capsize=2, )
            
            for j in range(3):
                x_ = [j-dodge, j, j+dodge]
                y = [pre_mean[j], post_mean[0, j], post_mean[-1, j]]
                ax.plot(x_, y, '-', color=plt.cm.plasma((j+1)/4))
                
                for m in range(5):
                    y = [pre[j, m], post[0, j, m], post[-1, j, m]]
                    ax.plot(x_, y, ':.', color=plt.cm.plasma((j+1)/4), alpha=.3)

            ax.set_xticks([0, 1, 2], ['0%->10%', '10%->30%', '30%->50%'])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if i == 0:
                ax.legend(frameon=False, loc=locs[i])
            ax.set_title(titles[i])
            ax.set_ylabel(r'$\mathcal{D}_{N/C}$')

        fig.savefig(os.path.join(self.fig5_path, 'fig5DEF.png'))

        # pearson corr
        ddc_all = [[ddc_o, ddc_pre_o], [ddc_i, ddc_pre_i]]

        dodge = 0.3
        x = np.repeat(np.arange(3), self.num_imgs)
        # cols = ("#7E2F8E", "#0072BD", "#77AC30")

        fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 4))
        tts = [r'$L^T r$', r'$\tilde{L}^T \alpha$']
        for j, (ax, ddc_pair) in enumerate(zip(axs, ddc_all)):
            post, pre = ddc_pair
            y = np.concatenate([pre, post[:, 0], post[:, -1]], -1)
            rrs = np.zeros((50, 3))
            for nnd in range(50):
                for nl in range(3):
                    res = pearsonr(x, y[nnd, nl])
                    rrs[nnd, nl] = res.statistic
            # plot
            fmt = ['--', '-', ':']
            labels = ['0%-10%', '10%-30%', '30%-50%']
            x_plot = np.arange(0, 200, 4)+1
            for i in range(3):
                ax.plot(x_plot, rrs[:, i], fmt[i], label=labels[i], color=plt.cm.plasma((i+1)/4))
                ax.scatter([49], [rrs[12, i]], marker='o', edgecolor='white', facecolor='k', zorder=2)
            ax.axhline(y=0, xmin=0.04, xmax=0.96, linewidth=1, color='r')
            if j == 1:
                ax.legend(frameon=False, loc='upper right')
            ax.set_ylabel(r'$\mathcal{D}_{N/C}$ correlation')
            ax.set_xlabel('subspace dimension')
            ax.set_title(tts[j])
            ax.set_ylim(-0.84, 0.3)
            ax.set_yticks([-0.6, -0.2, 0.2], [-0.6, -0.2, 0.2])

            ax.spines['left'].set_position(('data', 0))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        
        fig.savefig(os.path.join(self.figS4_path, 'figS4DE.png'))


if __name__ == '__main__':
    
    ## params
    def parse_args():
        parser = argparse.ArgumentParser("Fig5, S4")
        parser.add_argument("--input_path", type=str, required=True)
        parser.add_argument("--pre_response_path", type=str, required=True)
        parser.add_argument("--post_response_path", type=str, required=True)
        parser.add_argument("--jac_path", type=str, required=True)
        parser.add_argument("--proj_dist_path", type=str, required=True)

        return parser.parse_args()

    args = parse_args()

    plot = plot_mode(
        args.input_path,
        args.pre_response_path,
        args.post_response_path,
        args.jac_path,
        args.proj_dist_path
    )

    plot.Fig5AB()
    plot.Fig5C_S4BC()
    plot.Fig5DEF_S4DE()
