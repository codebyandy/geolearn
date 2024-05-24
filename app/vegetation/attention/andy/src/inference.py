import hydroDL.data.dbVeg
from hydroDL.data import dbVeg
import importlib
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import torch
from hydroDL.data import DataModel
from hydroDL.master import  dataTs2Range
from hydroDL import kPath
from sklearn.metrics import r2_score
import wandb
import pandas as pd
import argparse



def train(args, saveFolder):
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
    
    run_name = args.run_name
    # dataset = args.dataset
    rho = args.rho
    nh = args.nh

    epochs = args.epochs
    learning_rate = args.learning_rate
    nIterEp = args.iters_per_epoch
    sched_start_epoch = args.sched_start_epoch
    optimizer = args.optimizer
    global DROPOUT
    DROPOUT = args.dropout
    batch_size = args.batch_size
    test_epoch = args.test_epoch
    satellites = args.satellites

    if not args.testing:
        wandb.init(dir=os.path.join(kPath.dirVeg))
        wandb.run.name = run_name
        
    dataName = args.dataset
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
    
    # calculate position
    varS = ['VV', 'VH', 'vh_vv']
    varL = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ndvi', 'ndwi', 'nirv']
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

    model = 
    test(df, testInd, testIndBelow, ep, metrics)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    # admin
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--testing", type=bool, default=False)
    parser.add_argument("--cross_val", type=bool, default=False)
    parser.add_argument("--weights_path", type=str, default="")
    parser.add_argument("--device", type=int, default=-1)
    # dataset 
    parser.add_argument("--dataset", type=str, default="singleDaily",
                        choices=["singleDaily", "singleDaily-modisgrid", "singleDaily-nadgrid"])
    parser.add_argument("--rho", type=int, default=45)
    parser.add_argument("--satellites", type=str, default="all")
    # model
    parser.add_argument("--nh", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--dropout", type=float, default=0.1)
    # training
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--iters_per_epoch", type=int, default=20)
    parser.add_argument("--sched_start_epoch", type=int, default=200)
    parser.add_argument("--test_epoch", type=int, default=50)
    parser.add_argument("--sample", type=bool, default=False)
    args = parser.parse_args()
    
    # create save dir / save hyperparameters
    saveFolder = ""
    if not args.testing:
        saveFolder = os.path.join(kPath.dirVeg, 'runs', f"{args.run_name}")
        
        if not os.path.exists(saveFolder):
            os.mkdir(saveFolder)
        else:
            raise Exception("Run already exists!")
    
        json_fname = os.path.join(saveFolder, "hyperparameters.json")
        with open(json_fname, 'w') as f:
            tosave = vars(args)
            json.dump(tosave, f, indent=4)

    train(args, saveFolder)


