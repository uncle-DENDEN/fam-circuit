from utils import *
from model import ProtoRNN
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from pathlib import Path
import os.path as osp
import argparse

import matplotlib as mpl
from numpy.linalg import norm
from scipy.stats import ttest_rel, ttest_1samp, sem
import scipy.io as io
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.model_selection import train_test_split
from einops import rearrange, repeat
import seaborn as sns
import numpy as np
import torch

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

    new_path = path[:epoch_pos + 5] + str(epoch) + path[end_pos:]

    return new_path


## plot functions
class plot_fam_assoc:
    def __init__(self, input_path, pre_path, post_path, post_path_hr) -> None:
        
        ff_o = np.load(input_path)
        _, t_ind, n_ind = insertTeachingSig(ff_o, p=2)
        self.ff_o = rearrange(ff_o, 'n l p d h w -> (n l p) (d h w)')
    
        ys_pre = np.load(pre_path) 
        ys_pre = seqUnmix(ys_pre, t_ind, n_ind, n_imgs=5, npa=10)[..., :f.N_e] 
        ys_pre = f.r_numpy(ys_pre)[:, -2:].mean(-2)

        ys_all = [ys_pre]
        del ys_pre
        for i in range(5):
            ys_e = np.load(replace_epoch(post_path, i)) 
            ys_e = seqUnmix(ys_e, t_ind, n_ind, n_imgs=5, npa=10)[..., :f.N_e] 
            ys_e = f.r_numpy(ys_e)[:, -2:].mean(-2)
            ys_all.append(ys_e)
            del ys_e

        self.ys_all_exc = np.stack(ys_all)
        
        ys_e_hr = np.load(post_path_hr)
        self.temp_res = ys_e_hr.shape[-2]
        ys_e_hr = seqUnmix(ys_e_hr, t_ind, n_ind, n_imgs=5, npa=10)[..., :f.N_e]
        self.ys_e_hr = f.r_numpy(ys_e_hr) 

        self.n_pc = 200
        self.n_d = 2

        # output_path
        self.fig4_path = 'figs/fig4'
        Path(self.fig4_path).mkdir(parents=True, exist_ok=True)
        self.figS3_path = 'figs/figS3'
        Path(self.figS3_path).mkdir(parents=True, exist_ok=True)
    
    def pca(self, ys, ff):
        pca = PCA(n_components=self.n_pc)

        ys_all_inp = ys.reshape(-1, f.N_e) 
        X_all = pca.fit_transform(ys_all_inp)
        X_all = X_all.reshape(6, 5, 4, 10, self.n_pc) 

        ff_r = pca.fit_transform(ff)
        ff = ff_r.reshape(5, 4, 10, self.n_pc)

        ff = ff[..., :self.n_pc]
        X = X_all[..., :self.n_pc] 

        return X, ff
    
    def mds(self, ys, ff):
        emb = MDS(n_components=self.n_d, normalized_stress='auto')

        X_vs = []
        for i in range(6):
            ys_all_inp = ys[i]  # (200, f.N_e)
            X_v = emb.fit_transform(ys_all_inp.astype(np.float64))
            X_v = X_v.reshape(5, 4, 10, self.n_d)
            X_vs.append(X_v)

        ff_v = emb.fit_transform(ff.astype(np.float64))
        ff_v = ff_v.reshape(5, 4, 10, self.n_d)

        return X_vs, ff_v
    
    def cross_img_normalization(self, dis, ref):
        day, nl, img, pc = dis.shape
        dis = dis.reshape(day, nl, img, 1, pc)  # nl, img, 1, pc
        ref = ref.reshape(day, 1, 1, img, pc)  # 1, 1, img, pc
        dist = norm(dis - ref, axis=-1)  # day, nl, img, img
        correct = np.diagonal(dist, axis1=-2, axis2=-1)
        incorrect = (dist.sum(-1) - correct) / (img-1)
        
        return correct / incorrect  # day, nl, img
    
    def FigS3G(self):
        pca = PCA(n_components=self.n_pc)
        pca.fit(self.ff_o)
        pcd = [pca.explained_variance_ratio_]

        for n, d in enumerate([0, 5]):
            pca = PCA(n_components=self.n_pc)
            pca.fit(self.ys_all_exc[d])
            pcd.append(pca.explained_variance_ratio_)

        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(4, 2))
        labels = ['input', 'pre', 'post']
        cols = ["coral", "coral", "cornflowerblue"]
        for d in [1, 2]:
            ax.plot(pcd[d][:25], 'o', markersize=6, mfc=cols[d], mec='white', label=f'{labels[d]}')
            
        ax.legend(frameon=False)

        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.set_ylabel('variance')
        ax.set_xlabel('PCs')
        
        fig.savefig(osp.join(self.figS3_path, 'figS3G.png'))

    def FigS3DE(self):
        X_vs, ff_v = self.mds(self.ys_all_exc, self.ff_o)
        
        # noise cone
        X_v = X_vs[-1].mean(0)  # 4, 10, 3
        X_v_mean = X_v.mean(1)  # 4, 3

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)

        for i in range(4):
            X_vi = X_v[i]
            ax.scatter(*X_vi.T, color=plt.cm.Blues(1-(i/4)), s=25, alpha=.5)
            cov = np.cov(X_vi, rowvar=False)
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            ell_novel = Ellipse(xy=(X_vi.mean(0)[0], X_vi.mean(0)[1]), 
                                width=lambda_[0]*2*2, height=lambda_[1]*2*2, angle=np.rad2deg(np.arccos(v[0, 0])),
                                ec=plt.cm.Blues(1-(i/4)), fc='none')
            ax.add_artist(ell_novel)
            ax.scatter(*X_v_mean[i], marker='+', s=100, color='tab:red', alpha=1-0.2*i)

        # ax.set_ylim((-7, 7))
        # ax.set_xlim((-7, 7))

        ax.grid(color='0.9')

        ax.set_yticks([-1.7, 0, 1.5], [-1.7, 0, 1.5])
        ax.set_xticks([-1.6, 0, 1.6], [-1.6, 0, 1.6])

        fig.savefig(osp.join(self.figS3_path, 'figS3D.png'))

        # concept cone
        X_v = X_vs[-1].mean(-2)  # 5, 4, 3
        X_v_mean = X_v.mean(0)  # 4, 3

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)

        for i in range(4):
            X_vi = X_v[:, i]
            ax.scatter(*X_vi.T, color=plt.cm.Oranges(1-(i/4)), s=25, alpha=.5)
            cov = np.cov(X_vi, rowvar=False)
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            ell_novel = Ellipse(xy=(X_vi.mean(0)[0], X_vi.mean(0)[1]), 
                                width=lambda_[0]*2*2, height=lambda_[1]*2*2, angle=np.rad2deg(np.arccos(v[0, 0])),
                                ec=plt.cm.Oranges(1-(i/4)), fc='none')
            ax.add_artist(ell_novel)
            ax.scatter(*X_v_mean[i], marker='+', s=100, color='tab:red', alpha=1-0.2*i)

        # ax.set_ylim((-7, 7))
        # ax.set_xlim((-7, 7))

        ax.grid(color='0.9')

        ax.set_yticks([-7, 0, 7], [-7, 0, 7])
        ax.set_xticks([-7, 0, 7], [-7, 0, 7])

        fig.savefig(osp.join(self.figS3_path, 'figS3E.png'))

    def Fig4EG(self):
        X, ff = self.pca(self.ys_all_exc, self.ff_o)

        ff = ff.transpose(1, 0, 2, 3) 
        target_ff = ff[0].mean(axis=(0, 1)).reshape(1, 1, 1, -1) 
        dis_ff = ff - target_ff
        ref_ff = dis_ff[[0]]

        fp = X.transpose(0, 2, 1, 3, 4)
        day_num = fp.shape[0]
        target = fp[:, 0].mean(axis=(1, 2)).reshape(day_num, 1, 1, 1, -1) 
        dis = fp - target 
        ref = dis[:, [0]] 

        # directional alignmengt
        cosine_dist = np.sum(dis*ref, axis=-1)/(norm(dis, axis=-1)*norm(ref, axis=-1))
        cosine_dist_ff = np.sum(dis_ff*ref_ff, axis=-1)/(norm(dis_ff, axis=-1)*norm(ref_ff, axis=-1))
        cosine_dist = cosine_dist.mean(-1) 
        cosine_dist_ff = cosine_dist_ff.mean(-1)

        # relative distance
        np_num = dis.shape[-2]
        eff_dist = []
        eff_dist_ff = []
        for i in range(np_num):
            eff_dist_i = self.cross_img_normalization(dis[:, :, :, i], ref[:, :, :, i]) 
            eff_dist_ff_i = self.cross_img_normalization(np.expand_dims(dis_ff[:, :, i], 0), np.expand_dims(ref_ff[:, :, i], 0))[0]
            eff_dist.append(eff_dist_i)
            eff_dist_ff.append(eff_dist_ff_i)
        eff_dist = np.stack(eff_dist).mean(0)
        eff_dist_ff = np.stack(eff_dist_ff).mean(0)

        # plot
        day_num, nl, n_img = cosine_dist.shape
        levels = (1, 2, 3)
        labels = ("10%", "30%", "50%")
        cols = ("#E38D26", "#308192", "#AED2E2")

        dodge = 0.23
        dodge2 = 0.15
        y_margin = 0.03
        fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(12, 4))
        for n, (i, labels) in enumerate(zip(levels, labels)):
            cds = eff_dist[-1, i]
            cds_pre = eff_dist[0, i]
            stat, p = ttest_rel(cds, cds_pre, alternative='less')
            text = '*' if p < 0.05 else 'ns'

            cds2 = cosine_dist[-1, i] 
            cds2_pre = cosine_dist[0, i] 
            stat, p = ttest_rel(cds2, cds2_pre, alternative='greater')
            text2 = '*' if p < 0.05 else 'ns'

            cds_mean = cds.mean(-1)
            cds_sem = cds.std()
            e1 = axs[0].errorbar(n+dodge, cds_mean, yerr=cds_sem, elinewidth=2.6, fmt='o',
                                 markeredgecolor='k', markerfacecolor=cols[n], ecolor='k', 
                                 markersize=12, label='after')
            
            cds_pre_mean = cds_pre.mean()
            cds_pre_sem = cds_pre.std()
            e2 = axs[0].errorbar(n-dodge, cds_pre_mean, yerr=cds_pre_sem, elinewidth=2.6, fmt='^', 
                                 markeredgecolor='k', markerfacecolor=cols[n], ecolor='k', 
                                 markersize=12, label='before')
            
            y_pos = max(cds.max(), cds_pre.max())
            axs[0].plot([n-dodge, n+dodge], [y_pos+y_margin, y_pos+y_margin], color='k', linewidth=2)
            axs[0].plot([n-dodge, n-dodge], [y_pos+y_margin-0.01, y_pos+y_margin], color='k', linewidth=2)
            axs[0].plot([n+dodge, n+dodge], [y_pos+y_margin-0.01, y_pos+y_margin], color='k', linewidth=2)
            axs[0].text(n, y_pos+y_margin+0.002, text, fontsize=20)

            cds2_mean = cds2.mean(-1)
            cds2_sem = cds2.std()
            f1 = axs[1].errorbar(n+dodge, cds2_mean, yerr=cds2_sem, elinewidth=2.6, fmt='o', 
                                 markeredgecolor='k', markerfacecolor=cols[n], ecolor='k', 
                                 markersize=12, label='after')
            
            cds2_pre_mean = cds2_pre.mean()
            cds2_pre_sem = cds2_pre.std()
            f2 = axs[1].errorbar(n-dodge, cds2_pre_mean, yerr=cds2_pre_sem, elinewidth=2.6, fmt='^', 
                                 markeredgecolor='k', markerfacecolor=cols[n], ecolor='k', 
                                 markersize=12, label='before')
            
            y_pos = max(cds2.max(), cds2_pre.max())
            axs[1].plot([n-dodge, n+dodge], [y_pos+y_margin, y_pos+y_margin], color='k', linewidth=2)
            axs[1].plot([n-dodge, n-dodge], [y_pos+y_margin-0.01, y_pos+y_margin], color='k', linewidth=2)
            axs[1].plot([n+dodge, n+dodge], [y_pos+y_margin-0.01, y_pos+y_margin], color='k', linewidth=2)
            axs[1].text(n, y_pos+y_margin+0.002, text2, fontsize=20)

            if n == 2:
                handle1 = [e1, e2]
                handle2 = [f1, f2]

            for j in range(n_img):
                axs[0].plot([n-dodge2, n+dodge2], [cds_pre[j], cds[j]], '-', color='grey')
                axs[1].plot([n-dodge2, n+dodge2], [cds2_pre[j], cds2[j]], '-', color='grey')

        axs[0].set_xticks([0, 1, 2], ['10%', '30%', '50%'])
        axs[0].set_yticks([0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8])
        axs[0].set_ylabel("relative distance", fontsize=18)
        axs[1].set_xticks([0, 1, 2], ['10%', '30%', '50%'])
        axs[1].set_yticks([0.5, 0.7, 0.9], [0.5, 0.7, 0.9])
        axs[1].set_ylabel("directional alignment", fontsize=18)
            

        axs[0].spines.right.set_visible(False)
        axs[0].spines.top.set_visible(False)
        axs[1].spines.right.set_visible(False)
        axs[1].spines.top.set_visible(False)

        axs[0].legend(handles=handle1, frameon=False, fontsize=15)
        axs[1].legend(handles=handle2, frameon=False, fontsize=15)
        axs[0].spines['bottom'].set_linewidth(1.5)
        axs[0].spines['left'].set_linewidth(1.5)
        axs[1].spines['bottom'].set_linewidth(1.5)
        axs[1].spines['left'].set_linewidth(1.5)

        fig.savefig(osp.join(self.fig4_path, 'Fig4EG.png'))   

    def Fig4B(self, ni=0):
        pca = PCA(n_components=3)
        traj_r = pca.fit_transform(self.ys_e_hr.reshape(-1, f.N_e))
        traj_r = traj_r.reshape(5, 4, 10, self.temp_res, -1)

        traj = traj_r[ni].mean(1)
        nl = traj.shape[0]
        fig = plt.figure(figsize=(5, 5), constrained_layout=True)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=15, azim=45, roll=0)
        labels = ['target', '10%', '30%', '50%']
        cols = ("#666666", "#E38D26", "#308192", "#AED2E2")
        for i, traj_ in enumerate(traj):
            ax.plot(*traj_.T, '-', color=cols[i], linewidth=1, label=labels[i])
            ax.scatter(*traj_[0].T, s=49, marker='x', color=cols[i])
            ax.scatter(*traj_[-1].T, s=49, marker='o', color=cols[i])
            
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

        ax.legend(frameon=False, fontsize=13)
        ax.set_xlabel('PC1', labelpad=-10)
        ax.set_ylabel('PC2', labelpad=-10)
        ax.set_zlabel('PC3', labelpad=-10)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.set_zticks([], [])

        ax.set_box_aspect([2,2,2])  # Make the plot cubic
        ax.set_position([0.1, 0.1, 0.9, 0.9])  # [left, bottom, width, height]

        fig.savefig(osp.join(self.fig4_path, 'fig4B.png'))


if __name__ == '__main__':

    ## params
    def parse_args():
        parser = argparse.ArgumentParser("Fig4, S3")
        parser.add_argument("--input_path", type=str, required=True)
        parser.add_argument("--pre_path", type=str, required=True)
        parser.add_argument("--post_path", type=str, required=True)
        parser.add_argument("--post_path_hr", type=str, required=True)

        return parser.parse_args()

    args = parse_args()

    plot = plot_fam_assoc(
        args.input_path, 
        args.pre_path, 
        args.post_path, 
        args.post_path_hr
        )
    
    # plot.Fig4B()
    # plot.Fig4EG()
    # plot.FigS3DE()
    plot.FigS3G()