import anndata
import numpy as np
import pandas as pd
import pyemd


def mclust_R(x, n_clusters, model='EEE', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(x), n_clusters, model)
    mclust_res = np.array(res[-2]).astype(int) - 1

    return mclust_res


def euclid_dist(t1, t2):
    sum = 0
    for i in range(t1.shape[0]):
        sum += (t1[i] - t2[i]) ** 2
    return np.sqrt(sum)


def pairwise_distance(X):
    n = X.shape[0]
    adj = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            adj[i][j] = euclid_dist(X[i], X[j])
    return adj


def calculate_adj_matrix(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True):
    # x,y,x_pixel, y_pixel are lists
    if histology:
        assert (x_pixel is not None) & (x_pixel is not None) & (image is not None)
        assert (len(x) == len(x_pixel)) & (len(y) == len(y_pixel))
        print("Calculateing adj matrix using histology image...")
        # beta to control the range of neighbourhood when calculate grey vale for one spot
        # alpha to control the color scale
        beta_half = round(beta / 2)
        g = []
        for i in range(len(x_pixel)):
            max_x = image.shape[0]
            max_y = image.shape[1]
            nbs = image[max(0, x_pixel[i] - beta_half):min(max_x, x_pixel[i] + beta_half + 1),
                  max(0, y_pixel[i] - beta_half):min(max_y, y_pixel[i] + beta_half + 1)]
            g.append(np.mean(np.mean(nbs, axis=0), axis=0))
        c0, c1, c2 = [], [], []
        for i in g:
            c0.append(i[0])
            c1.append(i[1])
            c2.append(i[2])
        c0 = np.array(c0)
        c1 = np.array(c1)
        c2 = np.array(c2)
        print("Var of c0,c1,c2 = ", np.var(c0), np.var(c1), np.var(c2))
        c3 = (c0 * np.var(c0) + c1 * np.var(c1) + c2 * np.var(c2)) / (np.var(c0) + np.var(c1) + np.var(c2))
        c4 = (c3 - np.mean(c3)) / np.std(c3)
        z_scale = np.max([np.std(x), np.std(y)]) * alpha
        z = c4 * z_scale
        z = z.tolist()
        print("Var of x,y,z = ", np.var(x), np.var(y), np.var(z))
        X = np.array([x, y, z]).T.astype(np.float32)
    else:
        print("Calculateing adj matrix using xy only...")
        X = np.array([x, y]).T.astype(np.float32)
    return pairwise_distance(X)


def refine(sample_id, pred, dis, shape="hexagon"):
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :][dis_df.loc[index, :] > 0]
        nbs = dis_tmp[0: num_nbs + 1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if self_pred in nbs_pred:
            if (v_c.loc[self_pred] < num_nbs / 2) and (np.max(v_c) > num_nbs / 2):
                refined_pred.append(v_c.idxmax())
            else:
                refined_pred.append(self_pred)
        else:
            refined_pred.append(v_c.idxmax())
    return refined_pred


def local_func(para):
    i, j, ME_mat, ground_distance_mat = para
    first_histogram = ME_mat[i, :]
    second_histogram = ME_mat[j, :]
    first_histogram = first_histogram / np.sum(first_histogram)
    second_histogram = second_histogram / np.sum(second_histogram)
    cur_dist = pyemd.emd(first_histogram, second_histogram, ground_distance_mat)
    return cur_dist


# 该函数，将矩阵转换为adata数据，以便后续调用scanpy进行下游任务分析
def matrix_to_adata(matrix, adata: anndata.AnnData) -> anndata.AnnData:
    """
    convert a matrix to adata object, by
     - duplicating the original var (per-gene) annotations and adding "_nbr"
     - keeping the obs (per-cell) annotations the same as original anndata that banksy matrix was computed from
    """
    var_nbrs = adata.var.copy()
    var_nbrs.index += "_nbr"
    nbr_bool = np.zeros((var_nbrs.shape[0]*2,), dtype=bool)
    nbr_bool[var_nbrs.shape[0]:] = True
    print("num_nbrs:", sum(nbr_bool))

    var_combined = pd.concat([adata.var, var_nbrs])
    var_combined["is_nbr"] = nbr_bool
    # 尽可能多的保留原始adata的信息
    return anndata.AnnData(matrix, obs=adata.obs, var=var_combined, uns=adata.uns, obsm=adata.obsm)

