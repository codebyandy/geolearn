
import hydroDL.data.dbVeg
from hydroDL.data import dbVeg
import importlib
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from hydroDL.data import DataModel
from hydroDL.master import  dataTs2Range
import torch.optim as optim
from hydroDL import kPath
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import r2_score
import wandb
import time
import pandas as pd
import argparse
import shutil




from torch import nn
import torch
import math

import pdb


class InputFeature(nn.Module):
    def __init__(self, nTup, nxc, nh):
        super().__init__()
        self.nh = nh
        self.lnXc = nn.Sequential(nn.Linear(nxc, nh), nn.ReLU(), nn.Linear(nh, nh))
        self.lnLst = nn.ModuleList()
        for n in nTup:
            self.lnLst.append(
                nn.Sequential(nn.Linear(n, nh), nn.ReLU(), nn.Linear(nh, nh))
            )

    def getPos(self, pos):
        nh = self.nh
        P = torch.zeros([pos.shape[0], pos.shape[1], nh], dtype=torch.float32)
        for i in range(int(nh / 2)):
            P[:, :, 2 * i] = torch.sin(pos / (i + 1) * torch.pi)
            P[:, :, 2 * i + 1] = torch.cos(pos / (i + 1) * torch.pi)
        return P

    def forward(self, xTup, pTup, xc):
        outLst = list()
        for k in range(len(xTup)):
            x = self.lnLst[k](xTup[k]) + self.getPos(pTup[k])
            outLst.append(x)
        outC = self.lnXc(xc)
        out = torch.cat(outLst + [outC[:, None, :]], dim=1)
        return out


class AttentionLayer(nn.Module):
    def __init__(self, nx, nh):
        super().__init__()
        self.nh = nh
        self.W_k = nn.Linear(nx, nh, bias=False)
        self.W_q = nn.Linear(nx, nh, bias=False)
        self.W_v = nn.Linear(nx, nh, bias=False)
        self.W_o = nn.Linear(nh, nh, bias=False)

    def forward(self, x):
        # pdb.set_trace()
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        d = q.shape[1]
        score = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)
        attention_mask = torch.ones(score.shape)
        attention = torch.softmax(score * attention_mask, dim=-1)
        out = torch.bmm(attention, v)
        out = self.W_o(out)
        return out


class PositionWiseFFN(nn.Module):
    def __init__(self, nh, ny):
        super().__init__()
        self.dense1 = nn.Linear(nh, nh)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(nh, ny)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class FinalModel(nn.Module):
    def __init__(self, nTup, nxc, nh):
        super().__init__()
        self.nTup = nTup
        self.nxc = nxc
        self.encoder = InputFeature(nTup, nxc, nh)
        self.atten = AttentionLayer(nh, nh)
        self.addnorm1 = AddNorm(nh, DROPOUT)
        self.addnorm2 = AddNorm(nh, DROPOUT)
        self.ffn1 = PositionWiseFFN(nh, nh)
        self.ffn2 = PositionWiseFFN(nh, 1)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, pos, xcT, lTup):
        xIn = self.encoder(x, pos, xcT)
        out = self.atten(xIn)
        out = self.addnorm1(xIn, out)
        out = self.ffn1(out)
        out = self.addnorm2(xIn, out)
        out = self.ffn2(out)
        out = out.squeeze(-1)
        k = 0
        temp = 0
        for i in lTup:
            temp = temp + out[:, k : i + k].mean(-1)
            k = k + i
        temp = temp + out[:, k:].mean(-1)
        return temp



def train(args):
    def randomSubset(opt='train', batch=1000, sample=False):
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
    


    def test(df, testInd, testIndBelow, ep, metric):
        # test
        model.eval()
        varS = ['VV', 'VH', 'vh_vv']
        varL = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ndvi', 'ndwi', 'nirv']
        # varM = ["mod_b{}".format(x) for x in range(1, 8)]
        # varM = ["myd_b{}".format(x) for x in range(1, 8)]
        # varM = ['Fpar', 'Lai']
        varM = ["MCD43A4_b{}".format(x) for x in range(1, 8)]
        iS = [df.varX.index(var) for var in varS]
        iL = [df.varX.index(var) for var in varL]
        iM = [df.varX.index(var) for var in varM]
        yOut = np.zeros(len(testInd))
        
        for k, ind in enumerate(testInd):
            k
            xS = x[pSLst[ind], ind, :][:, iS][None, ...]
            xL = x[pLLst[ind], ind, :][:, iL][None, ...]
            xM = x[pMLst[ind], ind, :][:, iM][None, ...]
            pS = (pSLst[ind][None, ...] - rho) / rho
            pL = (pLLst[ind][None, ...] - rho) / rho
            pM = (pMLst[ind][None, ...] - rho) / rho
            xcT = xc[ind][None, ...]
            xS = torch.from_numpy(xS).float()
            xL = torch.from_numpy(xL).float()
            xM = torch.from_numpy(xM).float()
            pS = torch.from_numpy(pS).float()
            pL = torch.from_numpy(pL).float()
            pM = torch.from_numpy(pM).float()
            xcT = torch.from_numpy(xcT).float()

            xTup, pTup, lTup = (), (), ()
            if satellites == "no_landsat":
                xTup = (xS, xM)
                pTup = (pS, pM)
                lTup = (xS.shape[1], xM.shape[1])
            else:
                xTup = (xS, xL, xM)
                pTup = (pS, pL, pM)
                lTup = (xS.shape[1], xL.shape[1], xM.shape[1])
            
            yP = model(xTup, pTup, xcT, lTup)
            
            yOut[k] = yP.detach().numpy()
        
        yT = yc[testInd, 0]    
        
        obs = dm.transOutY(yT[:, None])[:, 0]
        pred = dm.transOutY(yOut[:, None])[:, 0]
        
        print("ABOVE THRESH")
        
        rmse = np.sqrt(np.mean((obs - pred) ** 2))
        corrcoef = np.corrcoef(obs, pred)[0, 1]
        coef_det = r2_score(obs, pred)
        obs_quality = [rmse, corrcoef, coef_det]
        
        print("All obs")
        print("rmse", rmse)
        print("corrcoef", corrcoef)
        print("coef det", coef_det)
        print("")
        
        # to site
        tempS = jInd[testInd]
        tempT = iInd[testInd]
        testSite = np.unique(tempS)
        pdb.set_trace()
        siteLst = list()
        matResult = np.ndarray([len(testSite), 3])
        for i, k in enumerate(testSite):
            ind = np.where(tempS == k)[0]
            t = df.t[tempT[ind]]
            siteName = df.siteIdLst[k]
            siteLst.append([pred[ind], obs[ind], t])
            matResult[i, 0] = np.mean(pred[ind])
            matResult[i, 1] = np.mean(obs[ind])
            matResult[i, 2] = np.corrcoef(pred[ind], obs[ind])[0, 1]
        
        # mean   
        rmse = np.sqrt(np.mean((matResult[:, 0] - matResult[:, 1]) ** 2))
        corrcoef = np.corrcoef(matResult[:, 0], matResult[:, 1])[0, 1]
        coef_det = r2_score(matResult[:, 0], matResult[:, 1])
        site_quality = [rmse, corrcoef, coef_det]
        
        # rmse
        print("Obs mean per site")
        print("rmse",rmse)
        print("corrcoef", corrcoef)
        print("coef det", coef_det)
        print("")
        
        # anomoly
        fig, ax = plt.subplots(1, 1)
        aLst, bLst = list(), list()
        
        for site in siteLst:
            aLst.append(site[0] - np.mean(site[0]))
            bLst.append(site[1] - np.mean(site[1]))
        
        a, b = np.concatenate(aLst), np.concatenate(bLst)
        ax.plot(np.concatenate(aLst), np.concatenate(bLst), '.')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        vmin = np.min([xlim[0], ylim[0]])
        vmax = np.max([xlim[1], ylim[1]])
        _ = ax.plot([vmin, vmax], [vmin, vmax], 'r-')
        ax.set_title("Obs diff from site means (Quality sites)")
        fig.show()
        
        # rmse
        print("Obs diff from site means")
        rmse = np.sqrt(np.mean((a - b) ** 2))
        corrcoef = np.corrcoef(a,b)[0, 1]
        coef_det = r2_score(a, b)
        anomaly_quality = [rmse, corrcoef, coef_det]
        
        print("rmse", rmse)
        print("corrcoef", corrcoef)
        print("coef det", coef_det)
        print("")
        
        ### Below THRESHOLD ###
        testInd = testIndBelow
        
        # test
        model.eval()
        varS = ['VV', 'VH', 'vh_vv']
        varL = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ndvi', 'ndwi', 'nirv']
        # varM = ["mod_b{}".format(x) for x in range(1, 8)]
        # varM = ["myd_b{}".format(x) for x in range(1, 8)]
        # varM = ['Fpar', 'Lai']
        varM = ["MCD43A4_b{}".format(x) for x in range(1, 8)]
        iS = [df.varX.index(var) for var in varS]
        iL = [df.varX.index(var) for var in varL]
        iM = [df.varX.index(var) for var in varM]
        yOut = np.zeros(len(testInd))
        
        for k, ind in enumerate(testInd):
            k
            xS = x[pSLst[ind], ind, :][:, iS][None, ...]
            xL = x[pLLst[ind], ind, :][:, iL][None, ...]
            xM = x[pMLst[ind], ind, :][:, iM][None, ...]
            pS = (pSLst[ind][None, ...] - rho) / rho
            pL = (pLLst[ind][None, ...] - rho) / rho
            pM = (pMLst[ind][None, ...] - rho) / rho
            xcT = xc[ind][None, ...]
            xS = torch.from_numpy(xS).float()
            xL = torch.from_numpy(xL).float()
            xM = torch.from_numpy(xM).float()
            pS = torch.from_numpy(pS).float()
            pL = torch.from_numpy(pL).float()
            pM = torch.from_numpy(pM).float()
            xcT = torch.from_numpy(xcT).float()

            xTup, pTup, lTup = (), (), ()
            if satellites == "no_landsat":
                xTup = (xS, xM)
                pTup = (pS, pM)
                lTup = (xS.shape[1], xM.shape[1])
            else:
                xTup = (xS, xL, xM)
                pTup = (pS, pL, pM)
                lTup = (xS.shape[1], xL.shape[1], xM.shape[1])
            
            yP = model(xTup, pTup, xcT, lTup)
            yOut[k] = yP.detach().numpy()
        
        yT = yc[testInd, 0]
        
        obs = dm.transOutY(yT[:, None])[:, 0]
        pred = dm.transOutY(yOut[:, None])[:, 0]
        
        print("BELOW THRESH")
        
        rmse = np.sqrt(np.mean((obs - pred) ** 2))
        corrcoef = np.corrcoef(obs, pred)[0, 1]
        coef_det = r2_score(obs, pred)
        obs_poor = [rmse, corrcoef, coef_det]
        
        print("All obs")
        print("rmse", rmse)
        print("corrcoef", corrcoef)
        print("coef det", coef_det)
        print("")
        
        
        # to site
        tempS = jInd[testInd]
        tempT = iInd[testInd]
        testSite = np.unique(tempS)
        siteLst = list()
        matResult = np.ndarray([len(testSite), 3])
        for i, k in enumerate(testSite):
            ind = np.where(tempS == k)[0]
            t = df.t[tempT[ind]]
            siteName = df.siteIdLst[k]
            siteLst.append([pred[ind], obs[ind], t])
            matResult[i, 0] = np.mean(pred[ind])
            matResult[i, 1] = np.mean(obs[ind])
            matResult[i, 2] = np.corrcoef(pred[ind], obs[ind])[0, 1]
        
        # mean
        rmse = np.sqrt(np.mean((matResult[:, 0] - matResult[:, 1]) ** 2))
        corrcoef = np.corrcoef(matResult[:, 0], matResult[:, 1])[0, 1]
        coef_det = r2_score(matResult[:, 0], matResult[:, 1])
        site_poor = [rmse, corrcoef, coef_det]
        
        # rmse
        print("Obs mean per site")
        print("rmse", rmse)
        print("corrcoef", corrcoef)
        print("coef det", coef_det)
        print("")
        
        # anomoly
        fig, ax = plt.subplots(1, 1)
        aLst, bLst = list(), list()
        
        for site in siteLst:
            aLst.append(site[0] - np.mean(site[0]))
            bLst.append(site[1] - np.mean(site[1]))
        
        a, b = np.concatenate(aLst), np.concatenate(bLst)
        
        # rmse
        print("Obs diff from site means")
        rmse = np.sqrt(np.mean((a - b) ** 2))
        corrcoef = np.corrcoef(a,b)[0, 1]
        coef_det = r2_score(a, b)
        anomaly_poor = [rmse, corrcoef, coef_det]
        
        print("rmse", rmse)
        print("corrcoef", corrcoef)
        print("coef det", coef_det)
        print("")
        
        data = {
            "epoch": [ep],
            "qual_obs_rmse": [obs_quality[0]],
            "qual_obs_corrcoef": [obs_quality[1]],
            "qual_obs_coefdet": [obs_quality[2]],
            "qual_site_rmse": [site_quality[0]],
            "qual_site_corrcoef": [site_quality[1]],
            "qual_site_coefdet": [site_quality[2]],
            "qual_anomaly_rmse": [anomaly_quality[0]],
            "qual_anomaly_corrcoef": [anomaly_quality[1]],
            "qual_anomaly_coefdet": [anomaly_quality[2]],
            "poor_obs_rmse": [obs_poor[0]],
            "poor_obs_corrcoef": [obs_poor[1]],
            "poor_obs_coefdet": [obs_poor[2]], 
            "poor_site_rmse": [site_poor[0]],
            "poor_site_corrcoef": [site_poor[1]],
            "poor_site_coefdet": [site_poor[2]],
            "poor_anomaly_rmse": [anomaly_poor[0]],
            "poor_anomaly_corrcoef": [anomaly_poor[1]],
            "poor_anomaly_coefdet": [anomaly_poor[2]]
        }
        
        full_test_metrics = {(k, v[0]) for (k, v) in data.items()}
        metrics.update(full_test_metrics)
        
        new_metrics = pd.DataFrame(data)
        metrics_path = os.path.join(saveFolder, 'metrics.csv')
        if os.path.exists(metrics_path):
            prev_metrics = pd.read_csv(metrics_path)
            new_metrics = pd.concat([prev_metrics, new_metrics])
            
        new_metrics.to_csv(os.path.join(saveFolder, 'metrics.csv'), index=False)
        torch.save(model.state_dict(), os.path.join(saveFolder, f'model_ep{ep}.pth'))

    
    ### START TRAINING ###
    
    # run_name = args.run_name
    # # dataset = args.dataset
    # rho = args.rho
    # nh = args.nh

    # epochs = args.epochs
    # learning_rate = args.learning_rate
    # nIterEp = args.iters_per_epoch
    # sched_start_epoch = args.sched_start_epoch
    # optimizer = args.optimizer
    global DROPOUT
    DROPOUT = 0
    # batch_size = args.batch_size
    # test_epoch = args.test_epoch
    # satellites = args.satellites

    # if not args.testing:
    #     wandb.init(dir=os.path.join(kPath.dirVeg))
    #     wandb.run.name = run_name
        
    model_dir_path = os.path.join(kPath.dirVeg, 'runs', args.model_dir)
    hyperparameters_path = os.path.join(model_dir_path, 'hyperparameters.json')

    with open(hyperparameters_path, 'r') as j:
        hyperparameters = json.loads(j.read())
        dataset = hyperparameters['dataset']
        satellites = hyperparameters['satellites']
        nh = hyperparameters['nh']
        rho = hyperparameters['rho']
        
    dataName = dataset
    importlib.reload(hydroDL.data.dbVeg)
    df = dbVeg.DataFrameVeg(dataName)
    dm = DataModel(X=df.x, XC=df.xc, Y=df.y)
    siteIdLst = df.siteIdLst
    dm.trans(mtdDefault='minmax')
    dataTup = dm.getData()
    dataEnd, (iInd, jInd) = dataTs2Range(dataTup, rho, returnInd=True)
    x, xc, y, yc = dataEnd
    
    iInd = np.array(iInd)
    jInd = np.array(jInd)
    
    # np.nanmean(dm.x[:, :, 0])
    # np.nanmax(df.x[:, :, 2])
    
    # calculate position
    varS = ['VV', 'VH', 'vh_vv']
    varL = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ndvi', 'ndwi', 'nirv']
    # varM = ["mod_b{}".format(x) for x in range(1, 8)]
    # varM = ["myd_b{}".format(x) for x in range(1, 8)]
    # varM = ['Fpar', 'Lai']
    varM = ["MCD43A4_b{}".format(x) for x in range(1, 8)]

    iS = [df.varX.index(var) for var in varS]
    iL = [df.varX.index(var) for var in varL]
    iM = [df.varX.index(var) for var in varM]
    
    pSLst, pLLst, pMLst = list(), list(), list()
    ns = yc.shape[0]
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
    
    np.where(nMat == 0)
    np.sum((np.where(nMat == 0)[1]) == 0)
    
    indKeep = np.where((nMat > 0).all(axis=1))[0]
    x = x[:, indKeep, :]
    xc = xc[indKeep, :]
    yc = yc[indKeep, :]
    nMat = nMat[indKeep, :]
    pSLst = [pSLst[k] for k in indKeep]
    pLLst = [pLLst[k] for k in indKeep]
    pMLst = [pMLst[k] for k in indKeep]
    
    jInd = jInd[indKeep]
    siteIdLst = [siteIdLst[k] for k in jInd]
    
    # split train and test
    jSite, count = np.unique(jInd, return_counts=True)
    countAry = np.array([[x, y] for y, x in sorted(zip(count, jSite))])
    
  
    # save data
    dataFolder = os.path.join(kPath.dirVeg, 'model', 'attention', 'dataset')
    subsetFile = os.path.join(dataFolder, 'subset.json')

    with open(subsetFile) as json_file:
        dictSubset = json.load(json_file)
    print("loaded dictSubset")
    
    trainInd = dictSubset['trainInd_k05']
    testInd = dictSubset['testInd_k05']
    testIndBelow = dictSubset['testInd_underThresh']
    
    bS = 8
    bL = 6
    bM = 10

    xS, xL, xM, pS, pL, pM, xcT, yT = randomSubset(opt='train')

    nTup, lTup = (), ()
    if satellites == "no_landsat":
        print("no landsat model")
        nTup = (xS.shape[-1], xM.shape[-1])
        lTup = (xS.shape[1], xM.shape[1])
    else:
        nTup = (xS.shape[-1], xL.shape[-1], xM.shape[-1])
        lTup = (xS.shape[1], xL.shape[1], xM.shape[1])
    
    nxc = xc.shape[-1]
    model = FinalModel(nTup, nxc, nh)
    model_weights_path = os.path.join(model_dir_path, 'best_model.pth')
    model.load_state_dict(torch.load(model_weights_path))
    loss_fn = nn.L1Loss(reduction='mean')

    # if optimizer == "adam":
    #     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # else:
    #     optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
    # scheduler = lr_scheduler.LinearLR(
    #     optimizer, start_factor=1.0, end_factor=0.01, total_iters=800
    # )
    
    model.train()
    for ep in range(1):
        lossEp = 0
        metrics = {"train_loss" : 0, "train_RMSE" : 0, "train_rsq" : 0, "train_Rsq" : 0}
        # for i in range(nIterEp):
            # t0 = time.time()
            # xS, xL, xM, pS, pL, pM, xcT, yT = randomSubset(opt='train', batch=batch_size, sample=args.sample)
            # t1 = time.time()
            # model.zero_grad()

            # xTup, pTup = (), ()
            # if satellites == "no_landsat":
            #     xTup = (xS, xM)
            #     pTup = (pS, pM)
            # else:
            #     xTup = (xS, xL, xM)
            #     pTup = (pS, pL, pM)
            
            # yP = model(xTup, pTup, xcT, lTup)      
            # loss = loss_fn(yP, yT)
            # loss.backward()
            # t2 = time.time()
            # lossEp = lossEp + loss.item()
            # optimizer.step()
    
            # metrics["train_loss"] += loss.item()
            # with torch.no_grad():
            #     obs, pred = yP.detach().numpy(), yT.detach().numpy()
            #     rmse = np.sqrt(np.mean((obs - pred) ** 2))
            #     corrcoef = np.corrcoef(obs, pred)[0, 1]
            #     coef_det = r2_score(obs, pred)
            #     metrics["train_RMSE"]+= rmse
            #     metrics["train_rsq"] += corrcoef
            #     metrics["train_Rsq"] += coef_det
    
        # metrics = {metric : sum / nIterEp for metric, sum in metrics.items()}
        
        # optimizer.zero_grad()
        # xS, xL, xM, pS, pL, pM, xcT, yT = randomSubset(opt='test', batch=1000, sample=args.sample)

        # xTup, pTup = (), ()
        # if satellites == "no_landsat":
        #     xTup = (xS, xM)
        #     pTup = (pS, pM)
        # else:
        #     xTup = (xS, xL, xM)
        #     pTup = (pS, pL, pM)
        
        # yP = model(xTup, pTup, xcT, lTup)
        # loss = loss_fn(yP, yT)
    
        # metrics["test_loss"] = loss_fn(yP, yT).item()
        # obs, pred = yP.detach().numpy(), yT.detach().numpy()
        # rmse = np.sqrt(np.mean((obs - pred) ** 2))
        # corrcoef = np.corrcoef(obs, pred)[0, 1]
        # coef_det = r2_score(obs, pred)
        # with torch.no_grad():
        #     metrics["test_RMSE"]= rmse
        #     metrics["test_rsq"] = corrcoef
        #     metrics["test_Rsq"] = coef_det
        
        # if ep > sched_start_epoch:
        #     scheduler.step()
        # print(
        #     '{} {:.3f} {:.3f} {:.3f} time {:.2f} {:.2f}'.format(
        #         ep, lossEp / nIterEp, loss.item(), corrcoef, t1 - t0, t2 - t1
        #     )
        # )

        # if ep > 0 and ep % test_epoch == 0:
        test(df, testInd, testIndBelow, ep, metrics)

        # if not args.testing:
        #     wandb.log(metrics)

    # metrics_path = os.path.join(saveFolder, 'metrics.csv')
    # metrics = pd.read_csv(metrics_path)

    # best_metrics = metrics[metrics.qual_obs_coefdet == max(metrics.qual_obs_coefdet)]
    # best_metrics['run_name'] = [run_name]
    # best_metrics = best_metrics[['run_name'] + [x for x in best_metrics.columns if x != 'run_name']]
    
    # old_best_model_path = os.path.join(saveFolder, f'model_ep{int(best_metrics.iloc[0].epoch)}.pth')
    # new_best_model_path = os.path.join(saveFolder, 'best_model.pth')
    # shutil.copyfile(old_best_model_path, new_best_model_path)

    # all_metrics_path = os.path.join(kPath.dirVeg, 'runs', 'best_metrics_all_runs.csv')
    # if os.path.exists(all_metrics_path):
    #     all_metrics = pd.read_csv(all_metrics_path)
    #     all_metrics = pd.concat([all_metrics, best_metrics])
    #     all_metrics.to_csv(all_metrics_path, index=False)
    # else:
    #     best_metrics.to_csv(all_metrics_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    # admin
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--model_dir", type=str)
    # save
    parser.add_argument("--save_folder", type=str, default=kPath.dirVeg)
    args = parser.parse_args()

    train(args)