import numpy as np
from scipy.stats import norm
import scipy.io as scio
from einops import rearrange
import time
import torch

#  This function calculates the multiscale fluctuation-based dispersion entropy (MFDE) of a univariate signal x
# % Ref:
# % [1] H. Azami, S. Arnold, S. Sanei, Z. Chang, G. Sapiro, J. Escudero, and A. Gupta, "Multiscale Fluctuation-based
# % Dispersion Entropy and its Applications to Neurological Diseases", IEEE ACCESS, 2019.
# % [2] H. Azami, and J. Escudero, "Amplitude-and Fluctuation-Based Dispersion Entropy", Entropy, vol. 20, no. 3, p.210, 2018.

def FDispEn_NCDF_ms(x,m,nc,mu,sigma,tau):
    N = len(x)
    y = norm.cdf(x, mu, sigma)
    y = np.where(y == 1, 1 - 1e-10, y)
    y = np.where(y == 0, 1e-10, y)
    z = np.round(y * nc + 0.5)
    all_patterns = np.arange(1, 2 * nc)
    key = all_patterns.copy()

    N_PDE = (2 * nc - 1)

    ind = []
    for i_h in range(1, m + 1):
        start_index = (i_h - 1) * tau + 1
        end_index = N - (m - 1) * tau + (i_h - 1) * tau + 1
        ind.append(list(range(start_index, end_index)))

    ind=np.array(ind)
    result = np.empty_like(ind)
    for i in range(ind.shape[0]):
        for j in range(ind.shape[1]):
            result[i, j] = z[ind[i, j]-1]
    embd2 = result


    dembd2 = np.diff(embd2.T) + nc
    emb = np.zeros((N - (m - 1) * tau, 1))


    for i_e in range(m - 1, 0, -1):
        emb = dembd2[:, i_e - 1] * 100 ** (i_e - 1) + emb
    emb=emb[0,:]

    pdf = np.zeros(N_PDE)

    for id in range(N_PDE):
        indices = np.where(emb == key[id])
        pdf[id] = len(indices[0])

    npdf = pdf / (N - (m - 1) * tau)
    p = npdf[npdf != 0]
    Out_FDispEn = -np.sum(p * np.log(p))

    return Out_FDispEn

def FDispEn_NCDF(x,m,nc,tau):
    N = len(x)
    mu = np.mean(x)
    sigma = np.std(x)
    temp = norm.cdf(x, mu, sigma)

    data_min = min(temp)
    data_max = max(temp)

    y = [(x - data_min) / (data_max - data_min) for x in temp]
    y = np.where(y == 1, 1 - 1e-10, y)

    y = np.where(y == 0, 1e-10, y)
    z = np.round(y * nc + 0.5)
    all_patterns = np.arange(1, 2 * nc)

    N_PDE = (2 * nc - 1)

    key = all_patterns.copy()

    ind = []
    for i_h in range(1, m + 1):
        start_index = (i_h - 1) * tau + 1
        end_index = N - (m - 1) * tau + (i_h - 1) * tau + 1
        ind.append(list(range(start_index, end_index)))

    ind = np.array(ind)
    result = np.empty_like(ind)
    for i in range(ind.shape[0]):
        for j in range(ind.shape[1]):
            result[i, j] = z[ind[i, j] - 1]
    embd2 = result

    dembd2 = np.diff(embd2.T) + nc
    emb = np.zeros((N - (m - 1) * tau, 1))

    for i_e in range(m - 1, 0, -1):
        emb = dembd2[:, i_e - 1] * 100 ** (i_e - 1) + emb
    emb = emb[0, :]

    pdf = np.zeros(N_PDE)

    for id in range(N_PDE):
        indices = np.where(emb == key[id])
        pdf[id] = len(indices[0])

    npdf = pdf / (N - (m - 1) * tau)
    p = npdf[npdf != 0]
    Out_FDispEn = -np.sum(p * np.log(p))
    return Out_FDispEn

def Multi(Data,S):
    L=len(Data)
    J = L // S
    M_Data = [sum(Data[(i - 1) * S:i * S]) / S for i in range(1, J + 1)]
    return M_Data


def MFDE(x,m,c,tau,Scale):

    Out_MFDE = np.zeros(Scale)
    Out_MFDE[0]=FDispEn_NCDF(x,m,c,tau)
    mu =np.mean(x)
    sigma = np.std(x)
    for j in range(2, Scale+1):
        xs = Multi(x,j)
        Out_MFDE[j-1]=FDispEn_NCDF_ms(xs,m,c,mu,sigma,tau)
    return Out_MFDE