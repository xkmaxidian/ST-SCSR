import numpy as np
import torch
from torch import device
def soft_numpy(x, T):
    if np.sum(np.abs(T)) == 0.:
        y = x
    else:
        y = np.maximum(np.abs(x) - T, 0.)
        y = np.sign(x) * y
    return y

def ST_SCSR(x, mat_be, mat_Ge, mat_fe, mat_bs, mat_fs, k, max_epoch, cosine_sim_matrix):
    n = x.shape[1]
    m = x.shape[0]
    T1 = np.zeros((n, n))
    mu =50

    mat_z = cosine_sim_matrix
    J1 = mat_z.copy()
    def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    α=2
    β=1
    # mat_z = np.random.rand(n, n)
    D1 = np.diag(np.sum(k, axis=1))

    for i in range(max_epoch):
        mat_be = mat_be * ((x @ mat_fe.T @ mat_Ge.T + np.finfo(float).eps) /
                           (mat_be @ mat_Ge @ mat_fe @ mat_fe.T @ mat_Ge.T))
        # mat_be = normalize(mat_be)


        mat_Ge = mat_Ge * ((mat_be.T @ x @ mat_fe.T + np.finfo(float).eps + 1*mat_bs.T @ k @ mat_fs.T) /
                           (
                                       mat_be.T @ mat_be @ mat_Ge @ mat_fe @ mat_fe.T + 1*mat_bs.T @ mat_bs @ mat_Ge @ mat_fs @ mat_fs.T + np.finfo(float).eps))
        mat_Ge = normalize(mat_Ge)
        # mat_z = mat_z * ((α * mat_fe.T @ mat_fe +  β *mat_fs.T @ mat_fs + np.finfo(float).eps ) / (
        #             α * mat_fe.T @ mat_fe @ mat_z +  β *mat_fs.T @ mat_fs @ mat_z ))

        mat_fe = mat_fe * (
                (mat_Ge.T @ mat_be.T @ x + 0.5 * mat_fe @ k + np.finfo(float).eps+ α * mat_fe @ mat_z + α *  mat_fe @ mat_z.T) /
                (
                        mat_Ge.T @ mat_be.T @ mat_be @ mat_Ge @ mat_fe + α * mat_fe + α * mat_fe @ mat_z @ mat_z.T + 0.5 * mat_fe @ D1))
        # mat_fe = normalize(mat_fe)
        mat_bs = mat_bs * ((k @ mat_fs.T @ mat_Ge.T + np.finfo(float).eps) /
                           (mat_bs @ mat_Ge @ mat_fs @ mat_fs.T @ mat_Ge.T +np.finfo(float).eps))
        mat_bs = normalize(mat_bs)
        # mat_Gs = mat_Gs * ((mat_bs.T @ k @ mat_fs.T + mat_Ge + np.finfo(float).eps) /
        #                    (mat_bs.T @ mat_bs @ mat_Gs @ mat_fs @ mat_fs.T + mat_Gs))

        mat_fs = mat_fs * ((mat_Ge.T @ mat_bs.T @ k + np.finfo(float).eps + β * mat_fs @ mat_z +β * mat_fs @ mat_z.T) /
                           (
                                       mat_Ge.T @ mat_bs.T @ mat_bs @ mat_Ge @ mat_fs + β * mat_fs +β * mat_fs @ mat_z @ mat_z.T +np.finfo(float).eps))
        mat_fs = normalize(mat_fs)
        # mat_z = mat_z * ((α * mat_fe.T @ mat_fe + β * mat_fs.T @ mat_fs + np.finfo(float).eps) / (
        #                 α * mat_fe.T @ mat_fe @ mat_z +  β *mat_fs.T @ mat_fs @ mat_z ))
        mat_z = mat_z * ((α * mat_fe.T @ mat_fe + β * mat_fs.T @ mat_fs + np.finfo(float).eps + mu * (
                    J1 - np.diag(np.diag(J1))) - T1) / (
                                 α * mat_fe.T @ mat_fe @ mat_z + β * mat_fs.T @ mat_fs @ mat_z + mu * mat_z))
        # mat_z = normalize(mat_z)
        mat_z[np.isnan(mat_z)] = 0
        mat_z = mat_z - np.diag(np.diag(mat_z))
        # 计算 J1 矩阵
        J1 = np.array(soft_numpy(mat_z + T1 / mu, 0.01))
        J1 = J1 - np.diag(np.diag(J1))
        T1 = T1 + mu * (mat_z - J1)


        err1 = np.linalg.norm(x - mat_be @ mat_Ge @ mat_fe, 'fro')
        err2 = np.linalg.norm(k - mat_bs @ mat_Ge @ mat_fs, 'fro')
        err3 = np.linalg.norm(mat_fe - mat_fe @ mat_z, 'fro')
        err4 = np.linalg.norm(mat_fs - mat_fs @ mat_z, 'fro')

        print("##err1 ", i, "is: ", err1)
        print("##err2 ", i, "is: ", err2)
        print("##err3 ", i, "is: ", err3)
        print("##err4 ", i, "is: ", err4)



    mat_z = 0.5 * (np.abs(mat_z) + np.abs(mat_z.T))
    return [mat_fe, mat_Ge, mat_z]
