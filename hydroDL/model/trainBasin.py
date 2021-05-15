import numpy as np
import torch
import time
from hydroDL.model import rnn, crit
from hydroDL.data import transform


def subsetRandom(dataLst, batchSize, sizeLst, opt='Random', matB=None):
    """get a random subset of training data
    Arguments:
        dataLst {list} --  see trainModel [x,xc,y,yc]
        batchSize {list} -- [spatial batch size, temporal batch size]
        sizeLst {list} -- list of following: (default: {None})
            nx {int} -- # input time series (default: {None})
            nxc {int} -- # input constant (default: {None})
            ny {int} -- # target time series (default: {None})
            nyc {int} -- # target constant (default: {None})
            ns {int} -- # pixels / instances (default: {None})
            nt {int} -- # time steps (default: {None})
    Returns:
        [torch.Tensor torch.Tensor] -- training subset
    """
    [x, xc, y, yc] = dataLst
    [rho, nbatch] = batchSize
    [nx, nxc, ny, nyc, nt, ns] = sizeLst
    if opt == 'Random':
        iS = np.random.randint(0, ns, [nbatch])
        iT = np.random.randint(0, nt-rho, [nbatch])
    elif opt == 'Weight':
        matB = ~np.isnan(y[rho:, :, :]) if matB is None else matB
        iS, iT = randomWeight(matB, ns, nt, nbatch, rho)
    xTemp = np.full([rho, nbatch, nx], np.nan)
    xcTemp = np.full([rho, nbatch, nxc], np.nan)
    yTemp = np.full([rho, nbatch, ny], np.nan)
    ycTemp = np.full([rho, nbatch, nyc], np.nan)
    if x is not None:
        for k in range(nbatch):
            xTemp[:, k, :] = x[iT[k]+1:iT[k]+rho+1, iS[k], :]
    if y is not None:
        for k in range(nbatch):
            yTemp[:, k, :] = y[iT[k]+1:iT[k]+rho+1, iS[k], :]
    if xc is not None:
        xcTemp = np.tile(xc[iS, :], [rho, 1, 1])
    if yc is not None:
        ycTemp[-1, :, :] = yc[iS, :]
    xTensor = torch.from_numpy(np.concatenate(
        [xTemp, xcTemp], axis=-1)).float()
    yTensor = torch.from_numpy(np.concatenate(
        [yTemp, ycTemp], axis=-1)).float()
    if torch.cuda.is_available():
        xTensor = xTensor.cuda()
        yTensor = yTensor.cuda()
    return xTensor, yTensor


def randomWeight(matB, ns, nt, nbatch, rho):
    # random by weight that from number of observations
    s1 = np.sum(matB, axis=2)
    s2 = np.sum(matB, axis=(0, 2))
    wT = s1/s2
    wS = s2 / np.sum(s2)
    iS = np.random.choice(ns, nbatch, p=wS)
    iT = np.zeros(nbatch).astype(int)
    for k in range(nbatch):
        iT[k] = np.random.choice(nt-rho, p=wT[:, iS[k]])
    return iS, iT


def getSize(dataTup):
    # dataTup: [x,xc,y,yc]
    sizeLst = list()
    for data in dataTup:
        if data is None:
            sizeLst.append(0)
        else:
            sizeLst.append(data.shape[-1])
    [nx, nxc, ny, nyc] = sizeLst
    x = dataTup[0]
    nt = x.shape[0]
    ns = x.shape[1]
    return (nx, nxc, ny, nyc, nt, ns)


def dealNaN(dataTup, optNaN):
    # check if any nan
    rmLst = list()
    dataLst = list()
    for data, optN in zip(dataTup, optNaN):
        if data is not None:
            data[np.isinf(data)] = np.nan
            indNan = np.where(np.isnan(data))
            if len(indNan[0]) > 0:
                if optN == 1:
                    data[indNan] = -1
                    print('nan found and filled -1')
                elif optN == 2:
                    if data.ndim == 2:
                        rmLst.append(indNan[0])
                    if data.ndim == 3:
                        rmLst.append(np.unique(np.where(np.isnan(data))[1]))
        dataLst.append(data)
    if len(rmLst) > 0:
        rmAry = np.concatenate(rmLst)
        for k in range(len(dataLst)):
            dataLst[k] = np.delete(dataLst[k], rmAry, axis=dataLst[k].ndim-2)
        print('nan found and removed')

    return dataLst


def trainModel(dataLst, model, lossFun, optim, batchSize=[None, 100],
               nEp=100, cEp=0, logFile=None, optBatch='Weight', nIterEp=None):
    """[summary]    
    Arguments:
        dataLst {list} --  see trainModel [x,xc,y,yc]
            x {np.array} -- input time series of size [nt,np,nx]
            xc {np.array} -- input constant of size [np,nxc]
            y {np.array} -- target time series of size [nt,np,ny]
            yc {np.array} -- target constant (or last time step) of size [np,nyc]
        batchSize {list} -- [spatial batch size, temporal batch size, None for use all]
        model {[type]} -- [description]
        lossFun {[type]} -- [description]

    Keyword Arguments:
        batchSize {list} -- [description] (default: {[100, 365]})
        nEp {int} -- [number of epochs to run] (default: {100})
        cEp {int} -- [current epoch (only for print)] (default: {0})

    Returns:
        [type] -- [description]
    """
    sizeLst = getSize(dataLst)
    [nx, nxc, ny, nyc, nt, ns] = sizeLst
    rho, nbatch = batchSize
    if rho is None:
        rho = nt
    if nbatch is None:
        nbatch = ns
    batchSize = [rho, nbatch]

    # training
    matB = ~np.isnan(dataLst[2][rho:, :, :])
    if nIterEp is None:
        if optBatch == 'Random':
            if nbatch*rho > ns*nt:
                nIterEp = 1
            else:
                nIterEp = int(
                    np.ceil(np.log(0.01) / np.log(1 - nbatch*rho/ns/nt)))
        elif optBatch == 'Weight':
            nSample = np.sum(matB)
            if nbatch*ny > nSample:
                nIterEp = 1
            else:
                nIterEp = int(
                    np.ceil(np.log(0.01) / np.log(1 - nbatch*ny/nSample)))
    print('iter per epoch {}'.format(nIterEp), flush=True)
    lossEp = 0
    lossEpLst = list()
    t0 = time.time()
    model.train()
    model.zero_grad()
    if logFile is not None:
        log = open(logFile, 'a')
    for iEp in range(1, nEp + 1):
        lossEp = 0
        t0 = time.time()
        # somehow the first iteration always failed
        if iEp == 1:
            try:
                xT, yT = subsetRandom(dataLst, batchSize,
                                      sizeLst, opt='Weight')
                yP = model(xT)
            except:
                print('first iteration failed again for CUDNN_STATUS_EXECUTION_FAILED ')
        for iIter in range(nIterEp):
            xT, yT = subsetRandom(dataLst, batchSize,
                                  sizeLst, opt='Weight', matB=matB)
            yP = model(xT)
            if type(lossFun) is crit.RmseLoss2D:
                loss = lossFun(yP, yT[-1, :, :])
            else:
                loss = lossFun(yP, yT)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()
            # test for nans
            for name, par in model.named_parameters():
                if par.requires_grad:
                    if torch.any(torch.isnan(par)):
                        print('nan par in {} epoch {}'.format(
                            name, iEp+cEp), flush=True)
            model.zero_grad()
            lossEp = lossEp + loss.item()
            # except:
            #     print('iteration Failed: iter {} ep {}'.format(iIter, iEp+cEp))
        lossEp = lossEp / nIterEp
        ct = time.time() - t0
        logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(iEp+cEp, lossEp, ct)
        print(logStr, flush=True)
        # log.write(logStr+'\n')
        print(logStr, file=log, flush=True)
        lossEpLst.append(lossEp)

    log.close()
    return model, optim, lossEpLst


def testModel(model, x, xc, ny=None, batchSize=100):
    model.eval()
    nt, ns, nx = x.shape
    iS = np.arange(0, ns, batchSize)
    iE = np.append(iS[1:], ns)
    yLst = list()
    ycLst = list()
    if batchSize > ns:
        batchSize = ns
    for k in range(len(iS)):
        print('batch: '+str(k))
        if xc is not None:
            xT = torch.from_numpy(np.concatenate(
                [x[:, iS[k]:iE[k], :], np.tile(xc[iS[k]:iE[k], :], [nt, 1, 1])], axis=-1)).float()
        else:
            xT = torch.from_numpy(x[:, iS[k]:iE[k], :]).float()
        if torch.cuda.is_available():
            xT = xT.cuda()
            model = model.cuda()
        if k == 0:
            try:
                yT = model(xT)
            except:
                print('first iteration failed again')
        yT = model(xT)
        out = yT.detach().cpu().numpy()
        if ny is None:
            yLst.append(out)
        else:
            yLst.append(out[:, :, :ny])
            ycLst.append(out[-1, :, ny:])
    if ny is None:
        yOut = np.concatenate(yLst, axis=1)
        return yOut
    else:
        yOut = np.concatenate(yLst, axis=1)
        ycOut = np.concatenate(ycLst, axis=0)
        return yOut, ycOut


def saveModel(model, modelFile):
    torch.save(model, modelFile)
