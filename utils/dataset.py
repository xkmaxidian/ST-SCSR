import numpy as np
from copy import deepcopy

import pandas as pd
import torch
from torch.utils import data
import scanpy as sc
from sklearn.decomposition import PCA
from torchtoolbox.transform import Cutout
from torchvision import transforms
import cv2
from scipy.sparse import issparse


class Dataset(data.Dataset):
    def __init__(self, adata, gene_preprocess='pca', n_comp=512,
                 prob_mask=0.5, pct_mask=0.2, prob_noise=0.5, pct_noise=0.8, sigma_noise=0.5,
                 prob_swap=0.5, pct_swap=0.1, img_size=112, train=True):
        super(Dataset, self).__init__()
        full_image = cv2.imread('E:/datasets/DLPFC/151676/151676_full_image.tif')
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
        patches = []
        # 图像裁剪，这需要adata中带有spatial信息
        for x, y in adata.obsm['spatial']:
            patches.append(full_image[y - img_size: y + img_size, x - img_size: x + img_size])
        patches = np.array(patches)

        # preprocess
        if gene_preprocess == 'pca':
            # sc.pp.pca(adata, n_comps=n_comp)
            if issparse(adata.X):
                pca = PCA(n_components=n_comp, random_state=42)
                self.gene = pca.fit_transform(adata.X.A)
                # self.gene = adata.X.A
            else:
                pca = PCA(n_components=n_comp, random_state=42)
                self.gene = pca.fit_transform(adata.X)
                # self.gene = adata.X
        elif gene_preprocess == 'hvg':
            self.gene = adata.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)

        self.image = patches

        self.label = adata.obs['Ground Truth']
        self.label = pd.Categorical(self.label).codes
        self.n_clusters = self.label.max() + 1
        self.spatial = adata.obsm['spatial']
        self.n_pos = self.spatial.max() + 1

        self.train = train

        self.image_train_transform = transforms.Compose([
            Cutout(0.5),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.image_test_tranform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gene_train_transform = GeneTransforms(self.gene.shape[1], prob_mask=prob_mask, pct_mask=pct_mask,
                                                   prob_noise=prob_noise, pct_noise=pct_noise, sigma_noise=sigma_noise,
                                                   prob_swap=prob_swap, pct_swap=pct_swap)

    def __getitem__(self, idx):
        spatial = torch.from_numpy(self.spatial[idx])
        y = self.label[idx]
        if self.train:
            xg = self.gene[idx]
            xg_u = self.gene_train_transform(deepcopy(xg))
            xg_v = self.gene_train_transform(deepcopy(xg))

            xg = torch.from_numpy(xg)
            xg_u = torch.from_numpy(xg_u)
            xg_v = torch.from_numpy(xg_v)

            xi_u = self.image_train_transform(self.image[idx])
            xi_v = self.image_train_transform(self.image[idx])
            return xg, xg_u, xg_v, xi_u, xi_v, spatial, y, idx
        else:
            xg = self.gene[idx]
            xg = torch.from_numpy(xg)
            xi = self.image_test_tranform(self.image[idx])
            return xg, xi, spatial, y, idx

    def __len__(self):
        return len(self.label)


class GeneTransforms(torch.nn.Module):
    def __init__(self, n_genes,
                 prob_mask, pct_mask,
                 prob_noise, pct_noise, sigma_noise,
                 prob_swap, pct_swap):
        super(GeneTransforms, self).__init__()

        self.n_genes = n_genes
        self.prob_mask = prob_mask
        self.pct_mask = pct_mask
        self.prob_noise = prob_noise
        self.pct_noise = pct_noise
        self.sigma_noise = sigma_noise
        self.prob_swap = prob_swap
        self.pct_swap = pct_swap

    def build_mask(self, pct_mask):
        mask = np.concatenate([np.ones(int(self.n_genes * pct_mask), dtype=bool),
                               np.zeros(self.n_genes - int(self.n_genes * pct_mask), dtype=bool)])
        np.random.shuffle(mask)
        return mask

    def forward(self, xg):
        if np.random.uniform(0, 1) < self.prob_mask:
            mask = self.build_mask(self.pct_mask)
            xg[mask] = 0

        if np.random.uniform(0, 1) < self.prob_noise:
            mask = self.build_mask(self.pct_noise)
            noise = np.random.normal(0, self.sigma_noise, int(self.n_genes * self.pct_noise))
            xg[mask] += noise

        if np.random.uniform(0, 1) < self.prob_swap:
            swap_pairs = np.random.randint(self.n_genes, size=(int(self.n_genes * self.pct_swap / 2), 2))
            xg[swap_pairs[:, 0]], xg[swap_pairs[:, 1]] = xg[swap_pairs[:, 1]], xg[swap_pairs[:, 0]]

        return xg
