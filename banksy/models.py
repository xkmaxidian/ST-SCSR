import os
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchtoolbox.tools import mixup_data, cutmix_data, mixup_criterion
from torchvision.models import densenet121
from uti1l.cluster import mclust_R, refine
from uti1l.loss import NT_Xent


def LinearBlock(input_dim, output_dim, p_drop):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )


class SpaCLR(nn.Module):
    def __init__(self, gene_dims, image_dims, p_drop, n_pos, projection_dims=[64, 64]):
        super(SpaCLR, self).__init__()

        gene_dims.append(projection_dims[0])
        self.gene_encoder = nn.Sequential(OrderedDict([
            (f'gene_block{i + 1}', LinearBlock(gene_dims[i], gene_dims[i + 1], p_drop)) for i, _ in
            enumerate(gene_dims[:-1])
        ]))
        self.mse_loss = nn.MSELoss()
        self.gene_decoder = nn.Linear(projection_dims[0], gene_dims[0])

        self.image_encoder = densenet121(pretrained=True)
        n_features = self.image_encoder.classifier.in_features
        self.image_encoder.classifier = nn.Identity()
        self.x_embedding = nn.Embedding(n_pos, n_features)
        self.y_embedding = nn.Embedding(n_pos, n_features)

        image_dims[0] = n_features
        image_dims.append(projection_dims[0])
        self.image_linear = nn.Sequential(OrderedDict([
            (f'image_block{i+1}', LinearBlock(image_dims[i], image_dims[i+1], p_drop))
            for i, _ in enumerate(image_dims[:-1])
        ]))

        self.projector = nn.Sequential(
            nn.Linear(projection_dims[0], projection_dims[0]),
            nn.ReLU(),
            nn.Linear(projection_dims[0], projection_dims[1]),
        )

    def forward_image(self, xi):
        xi = self.image_encoder(xi)
        xi = self.image_linear(xi)
        hi = self.projector(xi)
        return xi, hi

    def forward_gene(self, xg):
        xg = self.gene_encoder(xg)
        hg = self.projector(xg)
        return xg, hg

    def forward(self, xg, xi, spatial):
        xg, hg = self.forward_gene(xg)
        xi, hi = self.forward_image(xi)
        return xg, xi, hg, hi

    def recon_loss(self, zg, xg):
        zg = self.gene_decoder(zg)
        return self.mse_loss(zg, xg)


class TrainerSpaCLR:
    def __init__(self, adata, n_clusters, network, optimizer, device='cuda'):
        self.n_clusters = n_clusters
        self.network = network
        self.optimizer = optimizer

        # self.train_writer = SummaryWriter(log_dir + '_train')
        # self.valid_writer = SummaryWriter(log_dir + '_valid')
        self.device = device

        self.sample_id = adata.obs.index.tolist()
        self.adj_2d = adata.obsm['adj']

        self.w_g2g = 0.1
        self.w_recon = 0

    def eval_mclust_refined_ari(self, label, z):
        # if z.shape[0] < 1000:
        #     num_nbs = 4
        # else:
        #     num_nbs = 24
        num_nbs = 6
        ari, preds = mclust_R(z, self.n_clusters)
        refined_preds = refine(sample_id=self.sample_id, pred=preds, dis=self.adj_2d, num_nbs=num_nbs)
        ari = adjusted_rand_score(label, refined_preds)
        return ari

    def train(self, trainloader, epoch):
        with tqdm(total=len(trainloader)) as t:
            self.network.train()
            train_loss = 0
            train_cnt = 0

            for i, batch in enumerate(trainloader):
                t.set_description(f'Epoch {epoch} train')

                self.optimizer.zero_grad()
                xg, xg_u, xg_v, xi_u, xi_v, spatial, y, _ = batch
                xg = xg.to(self.device)
                xg_u = xg_u.to(self.device)
                xg_v = xg_v.to(self.device)
                xi_u = xi_u.to(self.device)
                xi_v = xi_v.to(self.device)
                spatial = spatial.to(self.device)

                criterion = NT_Xent(xg.shape[0])

                xg, xi_a, xi_b, lam = mixup_data(xg, xi_u)
                zg, hg = self.network.forward_gene(xg)
                zi_a, hi_a = self.network.forward_image(xi_a)
                zi_b, hi_b = self.network.forward_image(xi_b)
                g2i_loss = mixup_criterion(criterion, hg, hi_a, hi_b, lam)

                xg_u, xg_a, xg_b, lam = mixup_data(xg_u, xg_v)
                zg_u, hg_u = self.network.forward_gene(xg_u)
                zg_a, hg_a = self.network.forward_gene(xg_a)
                zg_b, hg_b = self.network.forward_gene(xg_b)
                g2g_loss = mixup_criterion(criterion, hg_u, hg_a, hg_b, lam) * self.w_g2g

                recon_loss = self.network.recon_loss(zg, xg) * self.w_recon

                loss = g2i_loss + g2g_loss + recon_loss
                loss.backward()
                self.optimizer.step()

                train_cnt += 1
                train_loss += loss.item()

                t.set_postfix(loss=f'{(train_loss / train_cnt):.3f}',
                              g2g_loss=f'{g2g_loss.item():.3f}',
                              recon_loss=f'{recon_loss.item():.3f}')
                t.update(1)

            # self.train_writer.add_scalar('loss', (train_loss / train_cnt), epoch)
            # self.train_writer.flush()

    def valid(self, validloader, epoch=0):
        Xg = []
        Xi = []
        Y = []
        with torch.no_grad():
            with tqdm(total=len(validloader)) as t:
                self.network.eval()
                valid_loss = 0
                valid_cnt = 0

                for i, batch in enumerate(validloader):
                    xg, xi, spatial, y, _ = batch
                    xg = xg.to(self.device)
                    xi = xi.to(self.device)
                    spatial = spatial.to(self.device)

                    xg, xi, hg, hi = self.network(xg, xi, spatial)
                    criterion = NT_Xent(xg.shape[0])
                    # loss of gene to image
                    loss = criterion(hg, hi)

                    valid_cnt += 1
                    valid_loss += loss.item()

                    Xg.append(xg.detach().cpu().numpy())
                    Xi.append(hg.detach().cpu().numpy())
                    Y.append(y)

                    t.set_postfix(loss=f'{(valid_loss / valid_cnt):.3f}')
                    t.update(1)

                Xg = np.vstack(Xg)
                Xi = np.vstack(Xi)
                Y = np.concatenate(Y, 0)
                # print(valid_loss)
        return Xg, Xi, Y

    def fit(self, trainloader, epochs):
        self.network = self.network.to(self.device)

        for epoch in range(epochs):
            self.train(trainloader, epoch + 1)

    def encode(self, batch):
        xg, xi, spatial, y, _ = batch
        xg = xg.to(self.device)
        xi = xi.to(self.device)
        spatial = spatial.to(self.device)
        xg, xi, hg, hi = self.network(xg, xi, spatial)
        return xg + 0.1 * xi

    def save_model(self, ckpt_path):
        torch.save(self.network.state_dict(), ckpt_path)

    def load_model(self, ckpt_path):
        self.network.load_state_dict(torch.load(ckpt_path))


class AutoEncoder(nn.Module):
    def __init__(self, dims):
        super(AutoEncoder, self).__init__()
