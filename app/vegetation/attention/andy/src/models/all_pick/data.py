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


def randomSubset(data, trainInd, testInd, satellites, opt='train', batch=1000,):
    df, _, _, _, nMat, pSLst, pLLst, pMLst, x, rho, xc, yc = data
    
    # random sample within window
    varS = ['VV', 'VH', 'vh_vv']
    varL = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ndvi', 'ndwi', 'nirv']
    # varM = ["mod_b{}".format(x) for x in range(1, 8)]
    # varM = ["myd_b{}".format(x) for x in range(1, 8)]
    # varM = ['Fpar', 'Lai']
    varM = ["MCD43A4_b{}".format(x) for x in range(1, 8)]
    if opt == 'train':
        indSel = np.random.permutation(trainInd)[0:batch]
    else:
        indSel = testInd

    iS = [df.varX.index(var) for var in varS]
    iL = [df.varX.index(var) for var in varL]
    iM = [df.varX.index(var) for var in varM]
    ns = len(indSel)
    
    matS1 = x[:, indSel, :][:, :, iS]
    matL1 = x[:, indSel, :][:, :, iL]
    matM1 = x[:, indSel, :][:, :, iM]
    xS = np.swapaxes(matS1, 0, 1)
    xL = np.swapaxes(matL1, 0, 1)
    xM = np.swapaxes(matM1, 0, 1)

    maskS1 = ~np.isnan(xS[:, :, 0])
    maskL1 = ~np.isnan(xL[:, :, 0])
    maskM1 = ~np.isnan(xM[:, :, 0])
    xS[np.isnan(xS)] = 0
    xL[np.isnan(xL)] = 0
    xM[np.isnan(xM)] = 0

    if satellites == 'no_landsat':
        mask = np.concatenate((maskS1, maskM1, np.ones((maskS1.shape[0], 1))), axis=1)
    else: 
        mask = np.concatenate((maskS1, maskL1, maskM1, np.ones((maskS1.shape[0], 1))), axis=1)
    
    return (
        torch.tensor(xS, dtype=torch.float32),
        torch.tensor(xL, dtype=torch.float32),
        torch.tensor(xM, dtype=torch.float32),
        torch.tensor(xc[indSel, :], dtype=torch.float32),
        torch.tensor(yc[indSel, 0], dtype=torch.float32),
        torch.tensor(mask, dtype=torch.int)
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