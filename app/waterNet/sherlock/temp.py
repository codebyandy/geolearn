
import os
from hydroDL.model import trainBasin, crit
from hydroDL.data import dbBasin, gageII
import numpy as np
import torch
import pandas as pd
from hydroDL.model import waterNetGlobal
import importlib

importlib.reload(waterNetGlobal)
importlib.reload(crit)

dataName = 'QN90ref'
# dataName = 'temp'
DF = dbBasin.DataFrameBasin(dataName)
label = 'test'
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'LAI']
mtdX = ['skip' for k in range(4)]+['norm']
varY = ['runoff']
mtdY = ['skip']
varXC = gageII.varLstEx
# mtdXC = dbBasin.io.extractVarMtd(varXC)
# mtdXC = ['QT' for var in varXC]
mtdXC = ['QT' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

# train
trainSet = 'WYB09'
testSet = 'WYA09'
DM1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM1.trans(mtdX=mtdX, mtdXC=mtdXC)
dataTup1 = DM1.getData()
DM2 = dbBasin.DataModelBasin(
    DF, subset=testSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM2.borrowStat(DM1)
dataTup2 = DM2.getData()

# model
nh = 16
ng = len(varXC)
ns = len(DF.siteNoLst)

model = waterNetGlobal.WaterNet3(nh, 1, ng)
model = model.cuda()
# optim = torch.optim.RMSprop(model.parameters(), lr=0.1)
optim = torch.optim.Adam(model.parameters(), lr=0.000001)
# lossFun = torch.nn.MSELoss().cuda()
lossFun = crit.LogLoss2D().cuda()


sn = 1e-8
# random subset
ns = len(DF.siteNoLst)
batchSize = [1000, 100]
sizeLst = trainBasin.getSize(dataTup1)
[x, xc, y, yc] = dataTup1
[nx, nxc, ny, nyc, nt, ns] = sizeLst
model.train()
batchSize = [1000, 10]
[rho, nbatch] = batchSize

# nIterEp = int(np.ceil(np.log(0.01)/np.log(1 - nbatch*rho/2000/nt)))
nIterEp = int(np.ceil((ns*nt)/(nbatch*rho)))
lossLst = list()


for k in range(100):
    iS = np.random.randint(0, ns, [nbatch])
    # iS = np.arange(k*nbatch, (k+1)*nbatch)
    print(iS)
    iT = np.random.randint(0, nt-rho, [nbatch])
    xTemp = np.full([rho, nbatch, nx], np.nan)
    xcTemp = np.full([nbatch, nxc], np.nan)
    yTemp = np.full([rho, nbatch, ny], np.nan)
    ycTemp = np.full([nbatch, nyc], np.nan)
    if x is not None:
        for k in range(nbatch):
            xTemp[:, k, :] = x[iT[k]+1:iT[k]+rho+1, iS[k], :]
    if y is not None:
        for k in range(nbatch):
            yTemp[:, k, :] = y[iT[k]+1:iT[k]+rho+1, iS[k], :]
    if xc is not None:
        xcTemp = xc[iS, :]
    if yc is not None:
        ycTemp = yc[iS, :]
    xT = torch.from_numpy(xTemp).float().cuda()
    xcT = torch.from_numpy(xcTemp).float().cuda()
    yT = torch.from_numpy(yTemp).float().cuda()
    ycT = torch.from_numpy(ycTemp).float().cuda()
    model.zero_grad()
    yP = model(xT, xcT)
    w = model.fc(xcT)
    print(w.min(), w.max())
    optim.zero_grad()
    loss = lossFun(yP[:, :, None], yT)
    with torch.autograd.set_detect_anomaly(True):
        loss.backward()
    # loss.backward()
    print(loss)
    optim.step()
    # optim.state_dict()
