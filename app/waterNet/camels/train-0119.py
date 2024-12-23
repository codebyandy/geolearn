
import random
import os
from hydroDL.model import trainBasin, crit, waterNetTest
from hydroDL.data import dbBasin, gageII
import numpy as np
import torch
import pandas as pd
from hydroDL.model import waterNetTest, waterNet
import importlib
from hydroDL.utils import torchUtils


dataName = 'camelsK'
# dataName = 'temp'
DF = dbBasin.DataFrameBasin(dataName)
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
varY = ['runoff']
mtdY = ['skip']
varXC = gageII.varLstEx
# mtdXC = dbBasin.io.extractVarMtd(varXC)
# mtdXC = ['QT' for var in varXC]
mtdXC = ['QT' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

# train
trainSet = 'WY8095'
testSet = 'WY9510'
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

nr = 5
model = waterNet.WaterNet0119(nh, len(varXC), nr)
model = model.cuda()
# optim = torch.optim.RMSprop(model.parameters(), lr=0.1)
optim = torch.optim.Adam(model.parameters())
# lossFun = torch.nn.MSELoss().cuda()
lossFun = crit.LogLoss2D().cuda()


sn = 1e-8
# random subset
sizeLst = trainBasin.getSize(dataTup1)
[x, xc, y, yc] = dataTup1
[nx, nxc, ny, nyc, nt, ns] = sizeLst
model.train()
batchSize = [1000, 100]
[rho, nbatch] = batchSize

# nIterEp = int(np.ceil(np.log(0.01)/np.log(1 - nbatch*rho/2000/nt)))
nIterEp = int(np.ceil((ns*nt)/(nbatch*rho)))
# nIterEp = 1
lossLst = list()
saveDir = r'/scratch/users/kuaifang/temp/'
# saveDir = r'C:\Users\geofk\work\waterQuality\waterNet\modelTemp'
# torch.autograd.set_detect_anomaly(True)
for ep in range(1, 1001):
    for iter in range(nIterEp):
        [rho, nbatch] = batchSize
        iS = np.random.randint(0, ns, [nbatch])
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
        loss = lossFun(yP[:, :, None], yT[nr-1:, :, :])
        optim.zero_grad()
        loss.backward()
        optim.step()
        # torchUtils.ifNan(model)
        print(ep, iter, loss.item())
        lossLst.append(loss.item())
    if ep % 50 == 0:
        modelFile = os.path.join(
            saveDir, 'wn0119-{}-ep{}'.format(dataName, ep))
        torch.save(model.state_dict(), modelFile)

lossFile = os.path.join(saveDir, 'loss-{}'.format('wn0119'))
pd.DataFrame(lossLst).to_csv(lossFile, index=False, header=False)
