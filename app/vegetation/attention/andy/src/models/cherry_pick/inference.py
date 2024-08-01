"""
This file can get either train or test metrics given a transformer model.

Usage:
- Call the script with paths to a model to get train and test metrics.
- Use `train_metrics` or `test_metrics` to get only those metrics.
"""

# hydroDl module by Kuai Fang
from hydroDL import kPath

import numpy as np
import json
import os
import torch
from sklearn.metrics import r2_score
import pandas as pd
import argparse
import shutil

from model import FinalModel
from data import randomSubset, prepare_data
from utils import varS, varL, varM

import pdb


def get_metrics(data, indices, config):
    '''
    Given data, split info, and model info, return RMSE, correlation coefficent, and coefficient of determination
    for all observations, by site, and for anomalies (all observations from site mean).

    Args
    - data: 
    - indices:
    - config
    Returns
    - 
    '''
    df, dm, iInd, jInd, _, pSLst, pLLst, pMLst, x, rho, xc, yc = data
    model, satellites  = config["model"], config["satellites"]

    model.eval()
    iS = [df.varX.index(var) for var in varS]
    iL = [df.varX.index(var) for var in varL]
    iM = [df.varX.index(var) for var in varM]
    yOut = np.zeros(len(indices))
    
    for k, ind in enumerate(indices):
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
    
    yT = yc[indices, 0]      
    obs = dm.transOutY(yT[:, None])[:, 0]
    pred = dm.transOutY(yOut[:, None])[:, 0]
    
    # obs
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    corrcoef = np.corrcoef(obs, pred)[0, 1]
    coef_det = r2_score(obs, pred)
    obs_metrics = [rmse, corrcoef, coef_det]
    
    # site
    tempS = jInd[indices]
    tempT = iInd[indices]
    testSite = np.unique(tempS)

    siteLst = list()
    matResult = np.ndarray([len(testSite), 3])
    for i, k in enumerate(testSite):
        ind = np.where(tempS == k)[0]
        t = df.t[tempT[ind]]
        siteLst.append([pred[ind], obs[ind], t])
        matResult[i, 0] = np.mean(pred[ind])
        matResult[i, 1] = np.mean(obs[ind])
        matResult[i, 2] = np.corrcoef(pred[ind], obs[ind])[0, 1]
     
    rmse = np.sqrt(np.mean((matResult[:, 0] - matResult[:, 1]) ** 2))
    corrcoef = np.corrcoef(matResult[:, 0], matResult[:, 1])[0, 1]
    coef_det = r2_score(matResult[:, 0], matResult[:, 1])
    site_metrics = [rmse, corrcoef, coef_det]
    
    # anomaly
    aLst, bLst = list(), list()
    
    for site in siteLst:
        aLst.append(site[0] - np.mean(site[0]))
        bLst.append(site[1] - np.mean(site[1]))
    
    a, b = np.concatenate(aLst), np.concatenate(bLst)

    rmse = np.sqrt(np.mean((a - b) ** 2))
    corrcoef = np.corrcoef(a,b)[0, 1]
    coef_det = r2_score(a, b)
    anomaly_metrics = [rmse, corrcoef, coef_det]
    
    return (obs_metrics, site_metrics, anomaly_metrics)


def test_metrics(data, split_indices, config):
    # get metrics for both (1) qualtiy and (2) poor sites
    obs_quality, site_quality, anomaly_quality = get_metrics(data, split_indices["test_quality"], config)
    obs_poor, site_poor, anomaly_poor = get_metrics(data, split_indices["test_poor"], config)

    # reformat all metrics to single dictionary
    metric_names = ["rmse", "corrcoef", "coefdet"]

    metrics = {"epoch": [config["epoch"]]}
    metrics.update({f"qual_obs_{metric_names[i]}" : [obs_quality[i]] for i in range(3)})
    metrics.update({f"qual_site_{metric_names[i]}" : [site_quality[i]] for i in range(3)})
    metrics.update({f"qual_anomaly_{metric_names[i]}" : [anomaly_quality[i]] for i in range(3)})
    metrics.update({f"poor_obs_{metric_names[i]}" : [obs_poor[i]] for i in range(3)})
    metrics.update({f"poor_site_{metric_names[i]}" : [site_poor[i]] for i in range(3)})
    metrics.update({f"poor_anomaly_{metric_names[i]}" :[anomaly_poor[i]] for i in range(3)})

    return metrics


def train_metrics(data, split_indices, config):
    obs, site, anomaly= get_metrics(data, split_indices["train"], config)

    # reformat all metrics to single dictionary
    metric_names = ["rmse", "corrcoef", "coefdet"]

    metrics = {}
    metrics.update({f"train_obs_{metric_names[i]}" : [obs[i]] for i in range(3)})
    metrics.update({f"train_site_{metric_names[i]}" : [site[i]] for i in range(3)})
    metrics.update({f"train_anomaly_{metric_names[i]}" : [anomaly[i]] for i in range(3)})

    return metrics


def save_best_model(model_weights_path):
    metrics_path = os.path.join(model_weights_path, 'metrics.csv')
    metrics = pd.read_csv(metrics_path)
    best_metrics = metrics[metrics.qual_obs_coefdet == max(metrics.qual_obs_coefdet)]
    old_best_model_path = os.path.join(model_weights_path, f'model_ep{int(best_metrics.iloc[0].epoch)}.pth')
    new_best_model_path = os.path.join(model_weights_path, 'best_model_so_far.pth')
    shutil.copyfile(old_best_model_path, new_best_model_path)


def main(args):        
    model_dir_path = os.path.join(kPath.dirVeg, 'runs', args.model_dir)
    hyperparameters_path = os.path.join(model_dir_path, 'hyperparameters.json')

    with open(hyperparameters_path, 'r') as j:
        hyperparameters = json.loads(j.read())
        dataset = hyperparameters['dataset']
        satellites = hyperparameters['satellites']
        nh = hyperparameters['nh']
        rho = hyperparameters['rho']

    data = prepare_data(dataset, rho)
    df, dm, iInd, jInd, nMat, pSLst, pLLst, pMLst, x, rho, xc, yc = data
    
    # get pre-created data splits
    dataFolder = os.path.join(kPath.dirVeg, 'model', 'attention', 'dataset')
    subsetFile = os.path.join(dataFolder, 'subset.json')

    with open(subsetFile) as json_file:
        dictSubset = json.load(json_file)
    print("loaded dictSubset")
    
    split_indices = {}
    split_indices["train"] = dictSubset['trainInd_k05']
    split_indices["test_quality"] = dictSubset['testInd_k05']
    split_indices["test_poor"] = dictSubset['testInd_underThresh']

    xS, xL, xM, pS, pL, pM, _, _ = randomSubset(data, split_indices["train"], split_indices["test_quality"], opt='train')

    nTup, lTup = (), ()
    if satellites == "no_landsat":
        print("no landsat model")
        nTup = (xS.shape[-1], xM.shape[-1])
        lTup = (xS.shape[1], xM.shape[1])
    else:
        nTup = (xS.shape[-1], xL.shape[-1], xM.shape[-1])
        lTup = (xS.shape[1], xL.shape[1], xM.shape[1])
    
    nxc = xc.shape[-1]
    model = FinalModel(nTup, nxc, nh, 0)
    
    model_weights_path = os.path.join(model_dir_path, 'best_model.pth')
    if not os.path.exists(model_weights_path): 
        save_best_model(model_dir_path)
        model_weights_path = os.path.join(model_dir_path, 'best_model_so_far.pth')
    model.load_state_dict(torch.load(model_weights_path))

    config = {"model" : model, "satellites" : satellites, "epoch" : None}
    metrics = {'model' : args.model_dir}
    metrics.update(test_metrics(data, split_indices, config))
    metrics.update(train_metrics(data, split_indices, config))

    metrics = pd.DataFrame(metrics)
    all_metrics_path = os.path.join(args.save_folder, 'inference.csv')
    if os.path.exists(all_metrics_path):
        all_metrics = pd.read_csv(all_metrics_path)
        all_metrics = pd.concat([all_metrics, metrics])
        all_metrics.to_csv(all_metrics_path, index=False)
    else:
        metrics.to_csv(all_metrics_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    # admin
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--model_dir", type=str)
    # save
    parser.add_argument("--save_folder", type=str, default=kPath.dirVeg)
    args = parser.parse_args()

    main(args)


