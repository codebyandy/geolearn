
import os
from hydroDL.model import trainBasin, crit, waterNetTestC, waterNetTest
from hydroDL.data import dbBasin, gageII, usgs
import numpy as np
import torch
import pandas as pd
import time

# extract data
dataName = 'weaG200All'
# def train(dataName, nm, codeLst):
DF = dbBasin.DataFrameBasin(dataName)
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
varY = ['runoff']+DF.varC
mtdY = ['skip'] + ['scale' for code in DF.varC]
varXC = gageII.varLstEx
mtdXC = ['QT' for var in varXC]
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
ng = xc.shape[-1]
nh = 16
nr = 5
nc = len(DF.varC)
nm = nh
[rho, nbatch] = batchSize

model = waterNetTestC.Wn0119EM(nh, ng, nr, nc, nm).cuda()
optim = torch.optim.Adam(model.parameters(), lr=0.01)
lossFun = crit.LogLoss3D().cuda()

# nIterEp = int(np.ceil(np.log(0.01)/np.log(1 - nbatch*rho/2000/nt)))
nIterEp = int(np.ceil((ns*nt)/(nbatch*rho)))
# nIterEp = 1
lossLst = list()
saveDir = r'/scratch/users/kuaifang/temp/'
# torch.autograd.set_detect_anomaly(True)
model.train()
isBroken = False
for ep in range(1, 1001):
    for iter in range(nIterEp):
        t0 = time.time()
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
        # if torch.isnan(yP).sum() > 0:
        #     isBroken = True
        #     print('break1')
        #     break
        loss = lossFun(yP[:, :, :], yT[nr-1:, :, :])
        # print('loss')
        lossQ = lossFun(yP[:, :, 0:1], yT[nr-1:, :, 0:1])
        lossC = lossFun(yP[:, :, 1:], yT[nr-1:, :, 1:])

        loss = lossQ*lossC
        optim.zero_grad()
        # print('backward')
        loss.backward()
        optim.step()
        printStr = '{} {} {:.3f} {:.3f} {:.2f}'.format(
            ep, iter, lossQ.item(), lossC .item(), time.time()-t0)
        print(printStr, flush=True)
        lossLst.append(loss.item())
    # if isBroken is True:
    #     print('break2')
    #     break
    if ep % 50 == 0:
        modelFile = os.path.join(
            saveDir, 'wnem0119-{}-ep{}'.format(dataName, ep))
        torch.save(model.state_dict(), modelFile)


lossFile = os.path.join(saveDir, 'loss-{}'.format(dataName))
# pd.DataFrame(lossLst).to_csv(lossFile, index=False, header=False)


# yOut, (QpR, QsR, QgR), (SfT, SsT, SgT), (cp,
#                                          cs, cg) = model(xT, xcT, outStep=True)
# torch.isnan(yOut).sum()
# torch.isnan(cp).sum()
# torch.isnan(cs).sum()
# torch.isnan(cg).sum()

# yP = model(xT, xcT)
# torch.isnan(yP).sum()

# # torch.isnan(yP).sum()


# # for j in range(yP.shape[1]):
# #     for i in range(yP.shape[2]):
# #         temp = torch.isnan(yP[:, j, i]).sum()
# #         if temp > 0:
# #             print(j, i, temp)


# # torch.isnan(xT[:, 76, :]).sum()
# # torch.isnan(xcT[76, :]).sum()
# # from hydroDL.model.waterNet import convTS, sepPar, WaterNet0119

# # aa = xcT[76:77, :]
# # c = model.fcC(xcT[76:77, :])
# # [cpT, csT, cgT] = sepPar(c, model.nm*model.nc, model.cLst)
# # cp = cpT.view(-1, nm, nc)
# # cs = csT.view(-1, nm, nc)
# # cg = cgT.view(-1, nm, nc)
# # cp = torch.relu(torch.exp(cp))
# # cs = torch.relu(torch.exp(cs))
# # cg = torch.relu(torch.exp(cg))
# # cp = cp.repeat(1, int(nh/nm), 1)
# # cs = cs.repeat(1, int(nh/nm), 1)
# # cg = cg.repeat(1, int(nh/nm), 1)


# # yOut, (QpR, QsR, QgR), (SfT, SsT, SgT), (cp, cs, cg)=model(xT,xcT,outStep=True)
# # torch.isnan(yOut).sum()
