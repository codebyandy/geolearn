import hydroDL.data.dbVeg
from hydroDL.data import dbVeg
from hydroDL.data import DataModel
from hydroDL.master import  dataTs2Range
from hydroDL import kPath

import numpy as np
import torch


# satellite variable names
varS = ['VV', 'VH', 'vh_vv']
varL = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ndvi', 'ndwi', 'nirv']
varM = ["MCD43A4_b{}".format(x) for x in range(1, 8)]

bS = 8
bL = 6
bM = 10


def randomSubset(data, trainInd, testInd, opt='train', batch=1000):
    df, _, _, _, nMat, pSLst, pLLst, pMLst, x, rho, xc, yc = data
    
    # random sample within window
    varS = ['VV', 'VH', 'vh_vv']
    varL = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ndvi', 'ndwi', 'nirv']
    varM = ["MCD43A4_b{}".format(x) for x in range(1, 8)]
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


def prepare_data(dataName, rho):
     # restructure data for training
    df = dbVeg.DataFrameVeg(dataName)
    dm = DataModel(X=df.x, XC=df.xc, Y=df.y)
    dm.trans(mtdDefault='minmax')
    dataTup = dm.getData()
    dataEnd, (iInd, jInd) = dataTs2Range(dataTup, rho, returnInd=True)
    x, xc, y, yc = dataEnd
    iInd = np.array(iInd)
    jInd = np.array(jInd)
    
    iS = [df.varX.index(var) for var in varS]
    iL = [df.varX.index(var) for var in varL]
    iM = [df.varX.index(var) for var in varM]
    
    pSLst, pLLst, pMLst = list(), list(), list()
    nMat = np.zeros([yc.shape[0], 3])
    for k in range(nMat.shape[0]):
        tempS = x[:, k, iS]
        pS = np.where(~np.isnan(tempS).any(axis=1))[0]
        tempL = x[:, k, iL]
        pL = np.where(~np.isnan(tempL).any(axis=1))[0]
        tempM = x[:, k, iM]
        pM = np.where(~np.isnan(tempM).any(axis=1))[0]
        pSLst.append(pS)
        pLLst.append(pL)
        pMLst.append(pM)
        nMat[k, :] = [len(pS), len(pL), len(pM)]
    
    indKeep = np.where((nMat > 0).all(axis=1))[0]
    x = x[:, indKeep, :]
    xc = xc[indKeep, :]
    yc = yc[indKeep, :]
    nMat = nMat[indKeep, :]
    pSLst = [pSLst[k] for k in indKeep]
    pLLst = [pLLst[k] for k in indKeep]
    pMLst = [pMLst[k] for k in indKeep]

    jInd = jInd[indKeep]
    data = (df, dm, iInd, jInd, nMat, pSLst, pLLst, pMLst, x, rho, xc, yc)
    return data