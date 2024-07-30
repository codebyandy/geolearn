"""
This file retrieves the daily LFMC predictions (from the 2016-10-15 to 2021-12-15) using the transformer model.
It outputs an array of shape (num observations, 3). Here, num observations is # total days * # of sites.
The columns are raw day (num days since 2016-10-15), site # (assigned by us), and LFMC prediction.

This is a fast temporary and hacky solution.

TO DO:
1. prepare_data_all outputs array taht stores the data inefficiently. Good for training, but high memory.
   Shape (rho, num obserations, num inputs). Lots of repeats.
2. Inference takes observations 1 at a time (because of varying shapes).
3. Exports raw day and site # (not interpretable unless converted).
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

# hydroDL module by Kuai Fang
import hydroDL.data.dbVeg
from hydroDL.data import dbVeg
from hydroDL.data import DataModel
from hydroDL.master import dataTs2Range
from hydroDL import kPath

import numpy as np
import torch

from utils import varS, varL, varM # Satellite variable names
from utils import bS, bL, bM # Days per satellite


def prepare_data_all(dataName, rho):
    """
    Loads the data, normalizes it, and processes it into the required shape and format 
    for further analysis. It identifies the available days for each observation from various remote 
    sensing sources (Sentinel, Landsat, MODIS) and filters the observations to retain only those with 
    data available from all sources.

    Args:
        dataName (str): The name or path of the dataset to be loaded.
        rho (int): The time window size for each observation.

    Returns:
        tuple: A tuple containing the following elements:
            - df (DataFrameVeg): A custom DataFrameVeg object containing the loaded data.
            - dm (DataModel): A custom DataModel object containing the normalized data.
            - iInd (np.array): Array of day indices for the observations.
            - jInd (np.array): Array of site indices for the observations.
            - nMat (np.array): Matrix indicating the number of available days with data for each satellite 
                               (Sentinel, Landsat, MODIS) for each observation.
            - pSLst (list): List of arrays indicating the days with available data for Sentinel for each observation.
            - pLLst (list): List of arrays indicating the days with available data for Landsat for each observation.
            - pMLst (list): List of arrays indicating the days with available data for MODIS for each observation.
            - x (np.array): Array of raw values for each observation in the shape (rho, number of observations, number of input features).
            - rho (int): The time window size for each observation.
            - xc (np.array): Array of constant variables for the observations.
            - yc (np.array): Array of LFMC data for the observations.
    """
    # Load data with custom DataFrameVeg and DataModel classes
    df = dbVeg.DataFrameVeg(dataName)
    dm = DataModel(X=df.x, XC=df.xc, Y=df.y)
    dm.trans(mtdDefault='minmax') # Min-max normalization
    dataTup = dm.getData()

    x, xc, y, _ = dataTup # NEW LINE
    # temp_y = np.zeros(y.shape) # TEMP LINE
    # temp_y[:2, :2, 0] = 1
    # dataTup = (x, xc, temp_y, _) # TEMP LINE
    dataTup = (x, xc, np.ones(y.shape), _) # NEW LINE

    # To convert data to shape (Number of observations, rho, number of input features)
    dataEnd, (iInd, jInd) = dataTs2Range(dataTup, rho, returnInd=True) # iInd: day, jInd: site
    x, xc, _, yc = dataEnd 
   
    iInd = np.array(iInd) # TODO: Temporary fix
    jInd = np.array(jInd) # TODO: Temporary fix
    
    iS = [df.varX.index(var) for var in varS]
    iL = [df.varX.index(var) for var in varL]
    iM = [df.varX.index(var) for var in varM]
    
    # For each remote sensing source (i.e. Sentinel, MODIS), for each LFMC observaton,
    # create a list of days in the rho-window that have data 
    # nMat: Number of days each satellite has data for, of shape (# obsevations, # satellites)
    pSLst, pLLst, pMLst = list(), list(), list()
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
    
    # only keep if data if there is at least 1 day of data for each remote sensing source
    indKeep = np.where((nMat > 0).all(axis=1))[0]
    x = x[:, indKeep, :]
    xc = xc[indKeep, :]
    yc = yc[indKeep, :]
    nMat = nMat[indKeep, :]
    pSLst = [pSLst[k] for k in indKeep]
    pLLst = [pLLst[k] for k in indKeep]
    pMLst = [pMLst[k] for k in indKeep]
    
    jInd = jInd[indKeep]
    iInd = iInd[indKeep]

    data = (df, dm, iInd, jInd, nMat, pSLst, pLLst, pMLst, x, rho, xc, yc)
    return data


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
        print(f"{k + 1} / {len(indices)} done")
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
    
    # yT = yc[indices, 0]      
    # obs = dm.transOutY(yT[:, None])[:, 0]
    pred = dm.transOutY(yOut[:, None])[:, 0]
    return pred



def main(args):        
    model_dir_path = os.path.join(kPath.dirVeg, 'runs', args.model_dir)
    hyperparameters_path = os.path.join(model_dir_path, 'hyperparameters.json')

    with open(hyperparameters_path, 'r') as j:
        hyperparameters = json.loads(j.read())
        dataset = hyperparameters['dataset']
        satellites = hyperparameters['satellites']
        nh = hyperparameters['nh']
        rho = hyperparameters['rho']

    data = prepare_data_all(dataset, rho)
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
    # if not os.path.exists(model_weights_path): 
    #     save_best_model(model_dir_path)
    #     model_weights_path = os.path.join(model_dir_path, 'best_model_so_far.pth')
    model.load_state_dict(torch.load(model_weights_path))
    config = {"model" : model, "satellites" : satellites, "epoch" : None}

    indices = [0, 1, 2, 3, 4, 5]
    pred = get_metrics(data, indices, config)
    jInd_ind = jInd[indices]
    iInd_ind = iInd[indices]

    pred = np.array(pred)
    jInd_ind = np.array(jInd_ind)
    iInd_ind = np.array(iInd_ind)

    res = np.column_stack([pred, jInd_ind, iInd_ind])
    path = os.path.join(kPath.dirVeg, "transformer_lfmc_daily.npy")
    np.save(path, res)
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    # admin
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--model_dir", type=str)
    # save
    parser.add_argument("--save_folder", type=str, default=kPath.dirVeg)
    args = parser.parse_args()

    main(args)


