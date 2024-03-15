
# custom packages
from hydroDL.data import dbVeg
from hydroDL.data import DataModel
from hydroDL.master import basinFull, slurm, dataTs2Range

import torch
import numpy as np


def get_shapes(data_tuple, rho, inputs):
    nh = 32
    xS, xL, xM, pS, pL, pM, xcT, yT = randomSubset(data_tuple, rho)
    df, trainInd, testInd, nMat, pSLst, pLLst, pMLst, x, xc, yc = data_tuple
    nxc = xc.shape[-1]
    
    if inputs == "no_M":
        nTup = (xS .shape[-1], xL.shape[-1])
        lTup = (xS.shape[1], xL.shape[1])
    elif inputs == "no_S":
        nTup = (xL.shape[-1], xM.shape[-1])
        lTup = (xL.shape[1], xM.shape[1])
    elif inputs == "no_L":
        nTup = (xS .shape[-1], xM.shape[-1])
        lTup = (xS.shape[1], xM.shape[1])
    else:
        nTup = (xS .shape[-1], xL.shape[-1], xM.shape[-1])
        lTup = (xS.shape[1], xL.shape[1], xM.shape[1])
    
    return nTup, nxc, lTup


def randomSubset(data_tuple, rho, opt='train', batch=1000):    
    df, trainInd, testInd, nMat, pSLst, pLLst, pMLst, x, xc, yc = data_tuple
    
    bS = 8
    bL = 6
    bM = 10
    
    # random sample within window
    varS = ['VV', 'VH', 'vh_vv']
    varL = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ndvi', 'ndwi', 'nirv']
    varM = ['Fpar', 'Lai']
    if opt == 'train':
        indSel = np.random.permutation(trainInd)[0:batch]
    else:
        indSel = testInd
    iS = [df.varX.index(var) for var in varS]
    iL = [df.varX.index(var) for var in varL]
    iM = [df.varX.index(var) for var in varM]
    ns = len(indSel)
    rS = np.random.randint(0, nMat[indSel, 0], [bS, ns]).T
    rL = np.random.randint(0, nMat[indSel, 1], [bL, ns]).T
    rM = np.random.randint(0, nMat[indSel, 2], [bM, ns]).T
    pS = np.stack([pSLst[indSel[k]][rS[k, :]] for k in range(ns)], axis=0)
    pL = np.stack([pLLst[indSel[k]][rL[k, :]] for k in range(ns)], axis=0)
    pM = np.stack([pMLst[indSel[k]][rM[k, :]] for k in range(ns)], axis=0)
    matS1 = x[:, indSel, :][:, :, iS]
    matL1 = x[:, indSel, :][:, :, iL]
    matM1 = x[:, indSel, :][:, :, iM]
    xS = np.stack([matS1[pS[k, :], k, :] for k in range(ns)], axis=0)
    xL = np.stack([matL1[pL[k, :], k, :] for k in range(ns)], axis=0)
    xM = np.stack([matM1[pM[k, :], k, :] for k in range(ns)], axis=0)
    pS = (pS - rho) / rho
    pL = (pL - rho) / rho
    pM = (pM - rho) / rho
    return (
        torch.tensor(xS, dtype=torch.float32),
        torch.tensor(xL, dtype=torch.float32),
        torch.tensor(xM, dtype=torch.float32),
        torch.tensor(pS, dtype=torch.float32),
        torch.tensor(pL, dtype=torch.float32),
        torch.tensor(pM, dtype=torch.float32),
        torch.tensor(xc[indSel, :], dtype=torch.float32),
        torch.tensor(yc[indSel, 0], dtype=torch.float32),
    )
