from multiprocessing import Pool

import networkx as nx
import numpy as np
import time
import seaborn as sns
import utils.cluster as clu
import scanpy as sc

from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import NearestNeighbors


def me_idx_me(I, cls_array, ME_var_names_np_unique):
    ME_X = np.zeros(shape=(I.shape[0], ME_var_names_np_unique.shape[0]))
    time_start = time.time()
    for i in range(I.shape[0]):

        if i % 1000 == 0:
            time_end = time.time()

            print('{0} MEs,time cost {1} s, {2} MEs, {3}s left'.format(i, time_end - time_start, I.shape[0] - i,
                                                                       (I.shape[0] - i) * (
                                                                               time_end - time_start) / 1000))
            time_start = time.time()

        cur_neighbors = I[i, :]

        cur_neighbors_cls = cls_array[cur_neighbors]
        cur_cls_unique, cur_cls_count = np.unique(cur_neighbors_cls, return_counts=True)  # counting for each cluster
        cur_cls_idx = [np.where(ME_var_names_np_unique == c)[0][0] for c in cur_cls_unique]  # c is string
        ME_X[i, cur_cls_idx] = cur_cls_count
    return ME_X


def MED(adata,
        use_cls,
        nn,
        ME_var_names_np_unique,
        copy=False,
        spatial_var='spatial',
        ME_key_added='ME',  # obsm key of ME
        MEidx_key_added='MEidx'  # uns key of ME_index

        ):
    if copy:
        adata_use = adata.copy()
    else:
        adata_use = adata

    ME_var_names_np = np.arange(ME_var_names_np_unique.shape[0]).astype('str')
    adata_use.obsm[ME_key_added] = np.zeros(shape=(adata_use.shape[0], ME_var_names_np.shape[0]))  # frequency
    time_start = time.time()
    spatial_mat = np.array(adata_use.obsm[spatial_var]).astype('int64')

    #     ##############exact knn search##############
    fit = NearestNeighbors(n_neighbors=nn).fit(spatial_mat)
    m = fit.kneighbors(spatial_mat)
    m = m[0], m[1]

    # sort_neighbors
    args = m[0].argsort(axis=1)
    add = np.arange(m[1].shape[0]) * m[1].shape[1]
    I = m[1].flatten()[args + add[:, None]]

    time_end = time.time()
    #     ##############knn end##############
    print('knn search time cost', time_end - time_start, 's')
    cls_array = adata_use.obs[use_cls]

    adata_use.uns[MEidx_key_added] = I

    ME_X = me_idx_me(I, cls_array, ME_var_names_np_unique)
    adata_use.obsm[ME_key_added] = ME_X

    if copy:
        return adata_use
    else:
        return ME_X


def get_cls_center(adata, cls_key, embed_key):
    cat = np.array(adata.obs[cls_key].cat.categories)
    embed_mat = np.array(adata.obsm[embed_key])
    cls_array = np.array(adata.obs[cls_key])

    mean_array = np.zeros(shape=(len(cat), embed_mat.shape[1]))
    for i in range(len(cat)):
        c = cat[i]
        mean_array[i] = np.mean(embed_mat[np.where(cls_array == c)])
    dist_mat = squareform(pdist(mean_array))
    return dist_mat


def get_ground_distance(adata, method='paga_guided_umap', cls_key=None, embed_key=None, connect_threshold=0.5):
    n_cls = np.array(adata.obs[cls_key].cat.categories).shape[0]
    if method == 'paga_graph':
        adj_mat_use = adata.uns['paga']['connectivities'].toarray()

        adj_mat_use = 1 / adj_mat_use

        np.fill_diagonal(adj_mat_use, 0)
        G = nx.from_numpy_matrix(adj_mat_use)

        len_path = dict(nx.all_pairs_dijkstra_path_length(G))
        ground_distance_mat = np.inf * np.ones(shape=(len(len_path), len(len_path)))

        for i in range(ground_distance_mat.shape[0]):
            for j in range(ground_distance_mat.shape[1]):
                ground_distance_mat[i, j] = len_path[i][j]
        np.fill_diagonal(ground_distance_mat, 0)
    elif method == 'paga':
        adj_mat_use = adata.uns['paga']['connectivities'].toarray()
        #         adj_mat_use[adj_mat_use==0] = -np.inf
        adj_mat_use = 1 - adj_mat_use
        np.fill_diagonal(adj_mat_use, 0)
        ground_distance_mat = adj_mat_use
    elif method == 'euclidean':  # lower bound of MED(approximation)
        ground_distance_mat = squareform(pdist(adata.uns['paga']['pos']))
    elif method == 'uniform':
        #         sz = adata.uns['paga']['pos'].shape[0]
        ground_distance_mat = np.ones(shape=(n_cls, n_cls))
        np.fill_diagonal(ground_distance_mat, 0)
    elif method == 'embed':
        #         cls_key = ''
        if cls_key is None or embed_key is None:
            print('need cls_ley or embed_key')
            return
        ground_distance_mat = get_cls_center(adata, cls_key, embed_key)
    elif method == 'paga_guided_umap':
        #         connect_threshold = 0.5

        embed_center = adata.uns['paga']['pos']
        embed_center_distmat = squareform(pdist(embed_center))

        adjacent_paga = adata.uns['paga']['connectivities'].toarray()
        adjacent_paga_bool = (adjacent_paga >= connect_threshold).astype('int')

        disconnected_cluster_idxs = np.where(np.sum(adjacent_paga_bool, axis=0) == 0)[0]
        for dis_idx in disconnected_cluster_idxs:
            cur_argmax = np.argmax(adjacent_paga[:, dis_idx])
            adjacent_paga_bool[dis_idx, cur_argmax] = 1
            adjacent_paga_bool[cur_argmax, dis_idx] = 1

        paga_guided_dist = np.multiply(embed_center_distmat, adjacent_paga_bool)
        G = nx.from_numpy_matrix(paga_guided_dist)
        ground_distance_mat = -np.inf * np.ones_like(paga_guided_dist)

        len_path = dict(nx.all_pairs_dijkstra_path_length(G))
        for i in len_path.keys():
            for j in len_path[i].keys():
                ground_distance_mat[i, j] = len_path[i][j]
        np.fill_diagonal(ground_distance_mat, 0)
        max_val = np.max(ground_distance_mat)
        np.nan_to_num(ground_distance_mat, copy=False, neginf=max_val * 2)

    adata.uns['GD'] = ground_distance_mat
    return ground_distance_mat


def plot_matrix(mat):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(mat, annot=True, ax=ax)
    plt.gca().set_aspect('equal', adjustable='box')


def med_ph_med_mp(adata,
                  GD_method='euclidean',
                  MED_knn=30,
                  CT_obs='clusters',
                  ifspatialplot=True,
                  OT_method='pyemd',
                  ME_precompyted=False,
                  GD_precomputed=False,
                  mp=100
                  ):
    # calculate ground distance, paga requried
    if not GD_precomputed:
        ground_distance_mat = get_ground_distance(adata, method=GD_method)
    else:
        ground_distance_mat = adata.uns['GD']
    # get ME histogram
    if not ME_precompyted:
        MED(adata, CT_obs, MED_knn)

    # calculate EMD matrix
    import time
    time_cost = 0
    time_start = time.time()
    ME_mat = adata.obsm['ME']
    ME_dist_EMD = np.zeros(shape=(ME_mat.shape[0], ME_mat.shape[0]))

    pool = Pool(mp)

    # prepare parameter list for multiprocessing
    para_list = []
    for i in range(ME_mat.shape[0]):
        for j in range(i, ME_mat.shape[0]):
            para_list.append((i, j, ME_mat, ground_distance_mat))
    # run mp
    rst_list = pool.map(clu.local_func, para_list)

    # fill matrix
    k = 0
    for i in range(ME_mat.shape[0]):
        for j in range(i, ME_mat.shape[0]):
            ME_dist_EMD[i, j] = rst_list[k]
            ME_dist_EMD[j, i] = rst_list[k]
            k += 1

    adata.obsm['X_ME_EMD_mat'] = ME_dist_EMD.copy()
    time_end = time.time()
    time_cost = time_end - time_start
    print(f'EMD distance matrix cost {time_cost}s')

    return adata


def merge_cls(old_cls, cls_i, cls_j):
    # cls_i = '3'
    # cls_j = '8'

    # old_cls = cur_ME_cls
    old_cls[old_cls == cls_i] = cls_j
    print(f'merged {cls_i} to {cls_j}')
    return old_cls


def merge_cls_paga(adata, cls_key='leiden_EMD', neighbor_key='neighbors_EMD', thresh=0.5, min_cls=2, paga_plot=True):
    if f'{cls_key}_merge_colors' in adata.uns:
        del adata.uns[f'{cls_key}_merge_colors']
    adata.obs[f'{cls_key}_merge'] = adata.obs[cls_key].copy()
    merge_cls_list = []
    while True:

        sc.tl.paga(adata, groups=f'{cls_key}_merge', neighbors_key=neighbor_key)
        if paga_plot:
            sc.pl.paga(adata, color=[f'{cls_key}_merge'], threshold=0)

        cur_conn = adata.uns['paga']['connectivities'].toarray()
        cur_ME_cls = adata.obs[f'{cls_key}_merge'].copy()
        if len(cur_ME_cls.cat.categories) <= min_cls:
            break
        merge_cls_idx = np.unravel_index(np.argmax(cur_conn), cur_conn.shape)
        # print(merge_cls_idx)
        if cur_conn[merge_cls_idx] < thresh:
            break
        merge_cls_i = cur_ME_cls.cat.categories[merge_cls_idx[0]]
        merge_cls_j = cur_ME_cls.cat.categories[merge_cls_idx[1]]

        new_ME_cls = merge_cls(cur_ME_cls, merge_cls_i, merge_cls_j)
        if paga_plot == True:
            del adata.uns[f'{cls_key}_merge_colors'][merge_cls_idx[0]]

        new_ME_cls = new_ME_cls.cat.remove_unused_categories()
        merge_cls_list.append(new_ME_cls.copy())
        adata.obs[f'{cls_key}_merge'] = new_ME_cls

    adata.uns['merge_cls_list'] = merge_cls_list
