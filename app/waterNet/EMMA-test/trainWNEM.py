
from sklearn.decomposition import PCA
import sklearn
import torch.nn.functional as F
import torch.nn as nn
import random
import os
from hydroDL.model import trainBasin, crit, waterNetTestC, waterNetTest
from hydroDL.data import dbBasin, gageII, usgs
import numpy as np
import torch
import pandas as pd
import importlib
from hydroDL.utils import torchUtils
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from hydroDL.model.waterNet import WaterNet0119, sepPar, convTS
from hydroDL import utils
importlib.reload(waterNetTestC)
# extract data
codeLst = ['00600', '00660', '00915', '00925', '00930', '00935', '00945']

siteNo = ['09163500', '04193500']
dataName = 'temp'
DF = dbBasin.DataFrameBasin.new(
    dataName, siteNo, varC=codeLst, varG=gageII.varLstEx)
DF.saveSubset('WYB09', sd='1982-01-01', ed='2009-10-01')
DF.saveSubset('WYA09', sd='2009-10-01', ed='2018-12-31')


DF = dbBasin.DataFrameBasin(dataName)
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
varY = ['runoff']+codeLst
mtdY = ['skip'] + ['scale' for code in codeLst]
varXC = gageII.varLstEx
mtdXC = ['skip' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

# train
trainSet = 'WYB09'
testSet = 'WYA09'
DM1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM1.trans(mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC)
dataTup1 = DM1.getData()
sizeLst = trainBasin.getSize(dataTup1)
[x, xc, y, yc] = dataTup1
[nx, nxc, ny, nyc, nt, ns] = sizeLst
batchSize = [1000, 100]
nh = 16
nr = 5
nm = nh
nc = len(codeLst)
model = waterNetTestC.Wn0119EM(nh, nr, nc, nm)
optim = torch.optim.Adam(model.parameters(), lr=0.01)
lossFun = crit.LogLoss3D().cuda()
# train
for ep in range(1, 101):
    [rho, nbatch] = batchSize
    iS = np.random.randint(0, ns, [nbatch])
    iT = np.random.randint(0, nt-rho, [nbatch])
    xTemp = np.full([rho, nbatch, nx], np.nan)
    yTemp = np.full([rho, nbatch, ny], np.nan)
    if x is not None:
        for k in range(nbatch):
            xTemp[:, k, :] = x[iT[k]+1:iT[k]+rho+1, iS[k], :]
    if y is not None:
        for k in range(nbatch):
            yTemp[:, k, :] = y[iT[k]+1:iT[k]+rho+1, iS[k], :]
    xT = torch.from_numpy(xTemp).float().cuda()
    yT = torch.from_numpy(yTemp).float().cuda()
    model.zero_grad()
    yP = model(xT)
    # loss = lossFun(yP[:, :, :], yT[nr-1:, :, :])
    lossQ = lossFun(yP[:, :, 0:1], yT[nr-1:, :, 0:1])
    lossC = lossFun(yP[:, :, 1:], yT[nr-1:, :, 1:])
    loss = lossQ*lossC
    optim.zero_grad()
    loss.backward()
    optim.step()
    # torchUtils.ifNan(model)
    print(ep, lossQ.item(), lossC.item())
# save model
saveDir = r'C:\Users\geofk\work\waterQuality\waterNet\modelTempEM'
modelFile = os.path.join(
    saveDir, 'wn0119-{}-ep{}-nm{}'.format(dataName, ep, nm))
torch.save(model.state_dict(), modelFile)
