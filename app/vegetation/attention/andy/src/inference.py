
from hydroDL import kPath
from hydroDL.data import dbVeg
from hydroDL.data import DataModel
from hydroDL.master import basinFull, slurm, dataTs2Range
from hydroDL.post import mapplot, axplot, figplot

from model import FinalModel
import data

import torch
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from sklearn.metrics import r2_score 
import os
import pickle
import pdb



def plot(obs, pred, ax):
    # fig, ax = plt.subplots(1, 1)
    ax.plot(pred, obs, '*')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    vmin = np.min([xlim[0], ylim[0]])
    vmax = np.max([xlim[1], ylim[1]])
    _ = ax.plot([vmin, vmax], [vmin, vmax], 'r-')
    return ax
    # return fig


def calc_metrics(obs, pred):
    RMSE = np.sqrt(np.mean((obs - pred) ** 2))
    rsq = np.corrcoef(pred, obs)[0][1] ** 2
    Rsq = r2_score(obs, pred)
    return RMSE, rsq, Rsq


def analysis(dict, rho=45, nh=32):
    with open(os.path.join(kPath.dirVeg, 'data_folds.pkl'), 'rb') as f:
        data_tuple = pickle.load(f)[0]
        
    metrics_data = ["RMSE", "rsq", "Rsq", 
            "site-mean RMSE", "site-mean rsq", "site-mean Rsq",
            "anomaly RMSE", "anomaly rsq", "anomaly Rsq"]
    metrics_data = {name : [] for name in metrics_data}
    final_fig, axes = plt.subplots(nrows=len(dict), ncols=3, figsize=(15, 3 * len(dict))) 
    
    for i, (run, (path, epoch)) in enumerate(dict.items()):
        # load model
        nTup, nxc, lTup = data.get_shapes(data_tuple, rho, "")  
        model = FinalModel(nTup, nxc, nh, "default")
        model_weights = os.path.join(kPath.dirVeg, "runs", path)
        model.load_state_dict(torch.load(model_weights))

        # inference
        _, (metrics, site_metrics, outlier_metrics) = inference(model, data_tuple, rho, "", axes[i])

        # metrics
        metrics_data["RMSE"].append(metrics[0])
        metrics_data["rsq"].append(metrics[1])
        metrics_data["Rsq"].append(metrics[2])
        metrics_data["site-mean RMSE"].append(site_metrics[0])
        metrics_data["site-mean rsq"].append(site_metrics[1])
        metrics_data["site-mean Rsq"].append(site_metrics[2])
        metrics_data["anomaly RMSE"].append(outlier_metrics[0])
        metrics_data["anomaly rsq"].append(outlier_metrics[1])
        metrics_data["anomaly Rsq"].append(outlier_metrics[2])

    # Set row names
    for i, row_name in enumerate(dict.keys()):
        axes[i, 0].set_ylabel(row_name, fontsize=16)

    axes[-1, 0].set_xlabel("All", fontsize=16)
    axes[-1, 1].set_xlabel("Site-Mean", fontsize=16)
    axes[-1, 2].set_xlabel("Anomaly", fontsize=16)
    
    final_metrics = pd.DataFrame(metrics_data)
    return final_metrics, final_fig
    

def inference(model, data_tuple, rho, inputs, axes=None, set="test"): 
    # set up data model
    df, trainInd, testInd, nMat, pSLst, pLLst, pMLst, x, xc, yc = data_tuple
    dm = DataModel(X=df.x, XC=df.xc, Y=df.y)
    dm.trans(mtdDefault='minmax') 

    selectedInd = testInd if set == "test" else trainInd

    # variables of interest
    varS = ['VV', 'VH', 'vh_vv']
    varL = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ndvi', 'ndwi', 'nirv']
    varM = ['Fpar', 'Lai']
    iS = [df.varX.index(var) for var in varS]
    iL = [df.varX.index(var) for var in varL]
    iM = [df.varX.index(var) for var in varM]
    yOut = np.zeros(len(selectedInd))

    model.eval()

    # iterate thru data, run predictions 
    for k, ind in enumerate(selectedInd):
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
        
        if inputs == "no_M":
            lTup = (xS.shape[1], xL.shape[1])
            yP = model((xS, xL), (pS, pL), xcT, lTup)
        elif inputs == "no_S":
            lTup = (xL.shape[1], xM.shape[1])
            yP = model((xL, xM), (pL, pM), xcT, lTup)
        elif inputs == "no_L":
            lTup = (xS.shape[1], xM.shape[1])
            yP = model((xS, xM), (pS, pM), xcT, lTup)
        else:
            lTup = (xS.shape[1], xL.shape[1], xM.shape[1])
            yP = model((xS, xL, xM), (pS, pL, pM), xcT, lTup)
        
        yOut[k] = yP.detach().numpy()

    yT = yc[selectedInd, 0]
    obs = dm.transOutY(yT[:, None])[:, 0]
    pred = dm.transOutY(yOut[:, None])[:, 0]
    
    metrics = calc_metrics(obs, pred)
    # if not axes:
    #     fig = plot(obs, pred, axes[0])

    # SITE MEAN
    dataTup = dm.getData()
    dataEnd, (iInd, jInd) = dataTs2Range(dataTup, rho, returnInd=True)
    tempS = jInd[selectedInd]
    tempT = iInd[selectedInd]
    site = np.unique(tempS)
    # siteLst = list()
    matResult = np.ndarray([len(site), 3])
    
    for i, k in enumerate(site):
        ind = np.where(tempS == k)[0]
        t = df.t[tempT[ind]]
        siteName = df.siteIdLst[k]
        # siteLst.append([pred[ind], obs[ind], t])
        matResult[i, 0] = np.mean(pred[ind])
        matResult[i, 1] = np.mean(obs[ind])
        matResult[i, 2] = np.corrcoef(pred[ind], obs[ind])[0, 1]

    site_metrics = calc_metrics(matResult[:, 0], matResult[:, 1])
    # if not axes:
    #     site_fig = plot(matResult[:, 0], matResult[:, 1], axes[1])
    
    # OUTLIER
    obs_dev_site_mean = []
    pred_dev_site_mean = []
    for i, k in enumerate(site):
        ind = np.where(tempS == k)[0]
        obs_dev_site_mean.append(obs[ind] - np.mean(obs[ind]))
        pred_dev_site_mean.append(pred[ind] - np.mean(pred[ind]))
    obs_dev_site_mean = np.concatenate(obs_dev_site_mean)
    pred_dev_site_mean = np.concatenate(pred_dev_site_mean)

    # outlier_fig = plot(obs_dev_site_mean, pred_dev_site_mean, axes[2])
    outlier_metrics = calc_metrics(obs_dev_site_mean, pred_dev_site_mean)


    return None, (metrics, site_metrics, outlier_metrics)


def create_map(run_info, rho=45, nh=32, inputs=""):
    with open(os.path.join(kPath.dirVeg, 'data_folds.pkl'), 'rb') as f:
        data_tuple = pickle.load(f)[0]
        
    path, epoch = run_info
 
    nTup, nxc, lTup = data.get_shapes(data_tuple, rho, "")  
    model = FinalModel(nTup, nxc, nh, "default")
    model_weights = os.path.join(kPath.dirVeg, "runs", path)
    model.load_state_dict(torch.load(model_weights))

    df, trainInd, testInd, nMat, pSLst, pLLst, pMLst, x, xc, yc = data_tuple
    dm = DataModel(X=df.x, XC=df.xc, Y=df.y)
    dm.trans(mtdDefault='minmax') 

    selectedInd = testInd

    # variables of interest
    varS = ['VV', 'VH', 'vh_vv']
    varL = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ndvi', 'ndwi', 'nirv']
    varM = ['Fpar', 'Lai']
    iS = [df.varX.index(var) for var in varS]
    iL = [df.varX.index(var) for var in varL]
    iM = [df.varX.index(var) for var in varM]
    yOut = np.zeros(len(selectedInd))

    model.eval()

    # iterate thru data, run predictions 
    for k, ind in enumerate(selectedInd):
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
        
        if inputs == "no_M":
            lTup = (xS.shape[1], xL.shape[1])
            yP = model((xS, xL), (pS, pL), xcT, lTup)
        elif inputs == "no_S":
            lTup = (xL.shape[1], xM.shape[1])
            yP = model((xL, xM), (pL, pM), xcT, lTup)
        elif inputs == "no_L":
            lTup = (xS.shape[1], xM.shape[1])
            yP = model((xS, xM), (pS, pM), xcT, lTup)
        else:
            lTup = (xS.shape[1], xL.shape[1], xM.shape[1])
            yP = model((xS, xL, xM), (pS, pL, pM), xcT, lTup)
        
        yOut[k] = yP.detach().numpy()

    yT = yc[selectedInd, 0]
    obs = dm.transOutY(yT[:, None])[:, 0]
    pred = dm.transOutY(yOut[:, None])[:, 0]
    
    # SITE MEAN
    dataTup = dm.getData()
    dataEnd, (iInd, jInd) = dataTs2Range(dataTup, rho, returnInd=True)
    tempS = jInd[selectedInd]
    tempT = iInd[selectedInd]
    site = np.unique(tempS)
    # siteLst = list()
    matResult = np.ndarray([len(site), 3])
    
    for i, k in enumerate(site):
        ind = np.where(tempS == k)[0]
        t = df.t[tempT[ind]]
        siteName = df.siteIdLst[k]
        # siteLst.append([pred[ind], obs[ind], t])
        matResult[i, 0] = np.mean(pred[ind])
        matResult[i, 1] = np.mean(obs[ind])
        matResult[i, 2] = np.corrcoef(pred[ind], obs[ind])[0, 1]

    pdb.set_trace()
    trainSite = np.unique(jInd[trainInd])
    lat = df.lat[trainSite]
    lon = df.lon[trainSite]
    figM = plt.figure(figsize=(8, 6))
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(
        figM, gsM[0, 0], lat, lon, np.zeros(len(lat)), cmap='gray', cb=False
    )

    testSite = np.unique(jInd[testInd])
    lat2 = df.lat[testSite]
    lon2 = df.lon[testSite]
    figM2 = plt.figure(figsize=(8, 6))
    gsM2 = gridspec.GridSpec(1, 1)
    axM2 = mapplot.mapPoint(figM2, gsM2[0, 0], lat2, lon2, matResult[:, 2], s=50)

    return figM, figM2