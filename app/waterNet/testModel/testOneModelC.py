
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
from hydroDL.model.waterNet import convTS, sepPar

from hydroDL.model import waterNetTest
import importlib

importlib.reload(waterNetTest)
importlib.reload(crit)

dataName = 'QN90ref'
DF = dbBasin.DataFrameBasin(dataName)

label = 'test'
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
# siteNo = '07148400'
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
nh = 8
nr = 3
model = waterNetTest.WaterNet0110(nh, len(varXC), nr)
model = model.cuda()

[x, xc, y, yc] = dataTup
xcP = torch.from_numpy(xc).float().cuda()

# one step

batchSize = [1000, 5]
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
xTrain = torch.from_numpy(xTemp).float().cuda()
xcTrain = torch.from_numpy(xcTemp).float().cuda()
yTrain = torch.from_numpy(yTemp).float().cuda()
ycTrain = torch.from_numpy(ycTemp).float().cuda()

nh = nh
ng = 55
nr = nr
fcR = nn.Sequential(
    nn.Linear(ng, 256),
    nn.Tanh(),
    nn.Dropout(),
    nn.Linear(256, nh*nr))
# [kp, ks, kg, gp, gl, qb, ga]
wLst = [
    'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid',
    'exp', 'relu', 'skip']
fcW = nn.Sequential(
    nn.Linear(ng, 256),
    nn.Tanh(),
    nn.Dropout(),
    nn.Linear(256, nh*len(wLst)))
# [vi,ve,vm]
vLst = ['skip', 'relu', 'exp']
fcT = nn.Sequential(
    nn.Linear(6+ng, 256),
    nn.Tanh(),
    nn.Dropout(),
    nn.Linear(256, nh*len(vLst)))
# [cs,cg,cb]
cLst = ['exp', 'exp', 'exp']
fcTC = nn.Sequential(
    nn.Linear(3+ng, 256),
    nn.Tanh(),
    nn.Dropout(),
    nn.Linear(256, nh*len(cLst)))
DP = nn.Dropout()
fcR = fcR.cuda()
fcW = fcW.cuda()
fcT = fcT.cuda()
fcTC = fcTC.cuda()


# forward
x = xTrain
xc = xcTrain
P, E, T1, T2, R, LAI = [x[:, :, k] for k in range(x.shape[-1])]
nt, ns = P.shape
nh = nh
nr = nr
Sf = torch.zeros(ns, nh).cuda()
Ss = torch.zeros(ns, nh).cuda()
Sg = torch.zeros(ns, nh).cuda()
xT = torch.cat([x, torch.tile(xc, [nt, 1, 1])], dim=-1)
w = fcW(xc)
[kp, ks, kg, gp, gL, qb, ga] = sepPar(w, nh, wLst)
gL = gL*2
kg = kg/10
ga = torch.softmax(DP(ga), dim=1)
v = fcT(xT)
[vi, ve, vm] = sepPar(v, nh, vLst)
vi = F.hardsigmoid(v[:, :, :nh]*2)
ve = ve*2
wR = fcR(xc)
vf = torch.arccos((T1+T2)/(T2-T1))/3.1415
vf[T1 >= 0] = 0
vf[T2 <= 0] = 1
Ps = P*vf
Pla = P*(1-vf)
Pl = Pla[:, :, None]*vi
Ev = E[:, :, None]*ve
Q1T = torch.zeros(nt, ns, nh).cuda()
Q2T = torch.zeros(nt, ns, nh).cuda()
Q3T = torch.zeros(nt, ns, nh).cuda()
for k in range(nt):
    qf = torch.minimum(Sf+Ps[k, :, None], vm[k, :, :])
    Sf = torch.relu(Sf+Ps[k, :, None]-vm[k, :, :])
    H = torch.relu(Ss+Pl[k, :, :]+qf-Ev[k, :, :])
    qp = torch.relu(kp*(H-gL))
    qs = ks*torch.minimum(H, gL)
    Ss = H-qp-qs
    qso = qs*(1-gp)
    qsg = qs*gp
    qg = kg*(Sg+qsg)+qb
    Sg = Sg-qg
    Q1T[k, :, :] = qp
    Q2T[k, :, :] = qso
    Q3T[k, :, :] = qg
r = torch.relu(wR[:, :nh*nr])
Q1R = convTS(Q1T, r)
Q2R = convTS(Q2T, r)
Q3R = convTS(Q3T, r)
outQ = torch.sum((Q1R+Q2R+Q3R)*ga, dim=2)

# concentrations
xTc = torch.cat([LAI[:, :, None], T1[:, :, None],
                T2[:, :, None], torch.tile(xc, [nt, 1, 1])], dim=-1)
wC = fcTC(xTc)
[cs, cg, cb] = sepPar(wC, nh, cLst)
C2T = Q2T*cs
C3T = Q2T*cg
C2R = convTS(C2T, r)
C3R = convTS(C3T, r)
outC = torch.sum((C2R+C3R)*ga, dim=2)/outQ
