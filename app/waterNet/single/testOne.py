
from hydroDL.model import trainBasin, crit
from hydroDL.data import dbBasin, gageII, gridMET
from hydroDL.master import basinFull
import numpy as np
from hydroDL import utils
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from hydroDL.model import waterNetTest
import importlib

importlib.reload(waterNetTest)
importlib.reload(crit)

dataName = 'QN90ref'
DF = dbBasin.DataFrameBasin(dataName)
DF.f[:, :, DF.varF.index('tmmn')] = DF.f[:, :, DF.varF.index('tmmn')] / 20
DF.f[:, :, DF.varF.index('tmmx')] = DF.f[:, :, DF.varF.index('tmmx')] / 20
label = 'test'
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(4)]+['norm' for k in range(2)]
varY = ['runoff']
mtdY = ['skip']
varXC = gageII.varLstEx
# mtdXC = dbBasin.io.extractVarMtd(varXC)
# mtdXC = ['QT' for var in varXC]
mtdXC = ['QT' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

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

# extract subset
siteNo = '03187500'
siteNoLst = DF.getSite(trainSet)
indS = siteNoLst.index(siteNo)
dataLst1 = list()
dataLst2 = list()
for dataLst, dataTup in zip([dataLst1, dataLst2], [dataTup1, dataTup2]):
    for data in dataTup:
        if data is not None:
            if data.ndim == 3:
                data = data[:, indS:indS+1, :]
            else:
                data = data[indS:indS+1, :]
        dataLst.append(data)
dataTup1 = tuple(dataLst1)
dataTup2 = tuple(dataLst2)

# model
nh = 16
# model = waterNetTest.WaterNet1115(nh, len(varXC))
model = waterNetTest.WaterNet1115(nh, len(varXC))
model = model.cuda()
# optim = torch.optim.RMSprop(model.parameters(), lr=0.1)
optim = torch.optim.Adam(model.parameters())
# optim = torch.optim.Rprop(model.parameters())
# lossFun = torch.nn.MSELoss().cuda()
lossFun = crit.LogLoss2D().cuda()

[x, xc, y, yc] = dataTup
xcP = torch.from_numpy(xc).float().cuda()

# random subset
model.train()
for kk in range(100):
    batchSize = [1000, 100]
    sizeLst = trainBasin.getSize(dataTup1)
    [x, xc, y, yc] = dataTup1
    [rho, nbatch] = batchSize
    [nx, nxc, ny, nyc, nt, ns] = sizeLst
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
    loss = lossFun(yP[:, :, None], yT)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(kk, loss.item())
    w = model.fc(xcT)
    # print(w[0, :])


model.eval()

t = DF.getT(trainSet)
[x, xc, y, yc] = dataTup

t = DF.getT(testSet)
[x, xc, y, yc] = dataTup2
xP = torch.from_numpy(x).float().cuda()
xcP = torch.from_numpy(xc).float().cuda()
yT = torch.from_numpy(y).float().cuda()
yOut, (q1Out, q2Out, q3Out) = model(xP, xcP, outQ=True)
yP = yOut.detach().cpu().numpy()
q1P = q1Out.detach().cpu().numpy()
q2P = q2Out.detach().cpu().numpy()
q3P = q3Out.detach().cpu().numpy()

lossFun(yOut[:, :, None], yT)
model.zero_grad()

k = 0
fig, ax = plt.subplots(1, 1)
ax.plot(t, yP[:, k], '-r')
ax.plot(t, y[:, k], '-k')
fig.show()

# REPEAT
x = xP
xc = xcP

P, E, T1, T2, R, LAI = [x[:, :, k] for k in range(6)]
nt, ns = P.shape
nh = model.nh
# Ta = (T1+T2)/2
vp = 1-torch.arccos((T1+T2)/(T2-T1))/3.1415
vp[T1 >= 0] = 1
vp[T2 <= 0] = 0
S0 = torch.zeros(ns, nh).cuda()
S1 = torch.zeros(ns, nh).cuda()
Sv = torch.zeros(ns, nh).cuda()
S2 = torch.zeros(ns, nh).cuda()
S3 = torch.zeros(ns, nh).cuda()
Yout = torch.zeros(nt, ns).cuda()
w = model.fc(xc)
xcT1 = torch.cat([LAI[:, :, None], torch.tile(xc, [nt, 1, 1])], dim=-1)
xcT2 = torch.cat([R[:, :, None], T1[:, :, None], T2[:, :, None],
                  torch.tile(xc, [nt, 1, 1])], dim=-1)
v1 = model.fcT1(xcT1)
v2 = model.fcT2(xcT2)
k1 = torch.sigmoid(w[:, nh:nh*2])
k2 = torch.sigmoid(w[:, nh*2:nh*3])
k23 = torch.sigmoid(w[:, nh*3:nh*4])
k3 = torch.sigmoid(w[:, nh*4:nh*5])/10
gl = torch.exp(w[:, nh*5:nh*6])*2
ga = torch.softmax(model.DP(w[:, nh*6:nh*7]), dim=1)
qb = torch.relu(w[:, nh*7:nh*8])
ge1 = torch.relu(w[:, nh*8:nh*9])
ge2 = torch.relu(w[:, nh*9:nh*10])
vi = F.hardsigmoid(v1[:, :, :nh])
vk = F.hardsigmoid(v1[:, :, nh:nh*2])
vm = torch.exp(v2[:, :, :nh]*2)
# vp = F.hardsigmoid(v2[:, :, -1])
Ps = P*(1-vp)
Pl = P*vp
Pl1 = Pl[:, :, None]*(1-vi)
Pl2 = Pl[:, :, None]*vi
Ev1 = E[:, :, None]*ge1
Ev2 = E[:, :, None]*ge2
Q1T = torch.zeros(nt, ns, nh).cuda()
Q2T = torch.zeros(nt, ns, nh).cuda()
Q3T = torch.zeros(nt, ns, nh).cuda()
for k in range(nt):
    H0 = S0+Ps[k, :, None]
    qSm = torch.minimum(H0, vm[k, :, :])
    Hv = torch.relu(Sv+Pl1[k, :, :] - Ev1[k, :, :])
    qv = Sv*vk[k, :, :]
    H2 = torch.relu(S2+qSm+qv-Ev2[k, :, :]+Pl2[k, :, :])
    Q1 = torch.relu(H2-gl)**k1
    q2 = torch.minimum(H2, gl)*k2
    Q2 = q2*(1-k23)
    H3 = S3+q2*k23
    Q3 = H3*k3+qb
    S0 = H0-qSm
    Sv = Hv-qv
    S2 = H2-Q1-q2
    S3 = H3-Q3
    Q1T[k, :, :] = Q1
    Q2T[k, :, :] = Q2
    Q3T[k, :, :] = Q3


# load LSTM
outName = '{}-{}'.format('QN90ref', trainSet)
yL, ycL = basinFull.testModel(
    outName, DF=DF, testSet=testSet, reTest=False, ep=1000)
yL = yL[:, indS, :]
yO = y[:, :, 0]
sd = 0
utils.stat.calNash(yL[sd:, :], yO[sd:, :])
utils.stat.calRmse(yL[sd:, :], yO[sd:, :])
utils.stat.calNash(yP[sd:, :], yO[sd:, :])
utils.stat.calRmse(yP[sd:, :], yO[sd:, :])

temp = vi.detach().cpu().numpy()[:, 0, :]
a = ga.detach().cpu().numpy()[0, :]
x = xP.detach().cpu().numpy()[:, 0, :]
fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(t, temp*a)
# axes[0].plot(t, temp)
axes[1].plot(t, yP, '-r')
axes[1].plot(t, yL, '-b')
# ax = axes[1].twinx()
# ax.plot(t, np.abs(yP-yO), '-r')
# ax.plot(t, np.abs(yL-yO), '-b')
axes[1].plot(t, y[:, k], '-k')
axes[2].plot(t, q1P[:, 0, :]*a)
fig.show()

fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(t, x[:,  [0, 2, 3, 4]])
axes[1].plot(t, yP, '-r')
axes[1].plot(t, yL, '-b')
axes[1].plot(t, y[:, k], '-k')
ax = axes[1].twinx()
ax.plot(t, np.abs(yP-yO)-np.abs(yL-yO), '--k')

axes[2].plot(t, np.abs(yP-yO), '-r')
axes[2].plot(t, np.abs(yL-yO), '-b')
fig.show()


T1 = -1
T2 = 30
1-np.arccos((T1+T2)/(T2-T1))/3.1415