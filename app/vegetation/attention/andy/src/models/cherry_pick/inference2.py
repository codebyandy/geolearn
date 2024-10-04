'''
This file retrieves the daily LFMC predictions (from the 2016-10-15 to 2021-12-15) using the transformer model.
It outputs an array of shape (num observations, 3). Here, num observations is # total days * # of sites.
The columns are raw day (num days since 2016-10-15), site # (assigned by us), and LFMC prediction.

This is a fast temporary and hacky solution.

TO DO:
1. prepare_data_all outputs array taht stores the data inefficiently. Good for training, but high memory.
   Shape (rho, num obserations, num inputs). Lots of repeats.
2. Inference takes observations 1 at a time (because of varying shapes).
3. Exports raw day and site # (not interpretable unless converted).
'''


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
    # Load data with custom DataFrameVeg and DataModel classes
    df = dbVeg.DataFrameVeg(dataName)
    dm = DataModel(X=df.x, XC=df.xc, Y=df.y)
    dm.trans(mtdDefault='minmax') # Min-max normalization
    dataTup = dm.getData()

    # NOTE: The 2 following lines distinguish prepare_data_all from data.prepare_data
    # dataTs2Range only keeps days with in-situ LFMC obsevation (i.e. where y is a value)
    # By making every y=1, we keep everyday
    x, xc, y, _ = dataTup # NEW LINE
    dataTup = (x, xc, np.ones(y.shape), _) # NEW LINE

    # To convert data to shape (Number of observations, rho, number of input features)
    dataEnd, (iInd, jInd) = dataTs2Range(dataTup, rho, returnInd=True) # iInd: day, jInd: site
    x, xc, _, yc = dataEnd 
   
    iInd = np.array(iInd) # TODO: Temporary fix
    jInd = np.array(jInd) # TODO: Temporary fix
    
    iS = [df.varX.index(var) for var in varS]
    iM = [df.varX.index(var) for var in varM]
    
    # For each remote sensing source (i.e. Sentinel, MODIS), for each LFMC observaton,
    # create a list of days in the rho-window that have data 
    # nMat: Number of days each satellite has data for, of shape (# obsevations, # satellites)
    pSLst, pMLst = list(), list(), list()
    nMat = np.zeros([yc.shape[0], 2])
    for k in range(nMat.shape[0]):
        tempS = x[:, k, iS]
        pS = np.where(~np.isnan(tempS).any(axis=1))[0]
        tempM = x[:, k, iM]
        pM = np.where(~np.isnan(tempM).any(axis=1))[0]
        pSLst.append(pS)

        pMLst.append(pM)
        nMat[k, :] = [len(pS), len(pM)]
    
    # only keep if data if there is at least 1 day of data for each remote sensing source
    indKeep = np.where((nMat > 0).all(axis=1))[0]
    x = x[:, indKeep, :]
    xc = xc[indKeep, :]
    yc = yc[indKeep, :]
    nMat = nMat[indKeep, :]
    pSLst = [pSLst[k] for k in indKeep]
    pMLst = [pMLst[k] for k in indKeep]
    
    jInd = jInd[indKeep]
    iInd = iInd[indKeep]

    data = {
        'dataFrame' : df,
        'dataModel' : dm,
        'day_indices' : iInd,
        'site_indices' : jInd,
        'sat_avail_per_obs' : nMat, 
        's_days_per_obs' : pSLst,
        'm_days_per_obs': pMLst,
        'rho' : rho,
        'x' : x,
        'xc' : xc,
        'yc' : yc
    }

    return data


def get_predictions(data, indices, model):
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
    df = data['dataFrame'] 
    dm = data['dataModel']
    iInd = data['day_indices']
    jInd = data['site_indices']
    pSLst = data['s_days_per_obs'] 
    pMLst = data['m_days_per_obs']
    rho = data['rho']
    x = data['x']
    xc = data['xc']
    yc = data['yc']
    
    model.eval()

    iS = [df.varX.index(var) for var in varS]
    iM = [df.varX.index(var) for var in varM]
    yOut = np.zeros(len(indices))
    
    for k, ind in enumerate(indices):
        print(f'{k + 1} / {len(indices)} done')
        xS = x[pSLst[ind], ind, :][:, iS][None, ...]
        xM = x[pMLst[ind], ind, :][:, iM][None, ...]
        pS = (pSLst[ind][None, ...] - rho) / rho
        pM = (pMLst[ind][None, ...] - rho) / rho
        xcT = xc[ind][None, ...]
        xS = torch.from_numpy(xS).float()
        xM = torch.from_numpy(xM).float()
        pS = torch.from_numpy(pS).float()
        pM = torch.from_numpy(pM).float()
        xcT = torch.from_numpy(xcT).float()

        xTup = (xS, xM)
        pTup = (pS, pM)
        lTup = (xS.shape[1], xM.shape[1])

        yP = model(xTup, pTup, xcT, lTup)    
        yOut[k] = yP.detach().numpy()

    yT = yc[indices, 0]      
    obs = dm.transOutY(yT[:, None])[:, 0]
    pred = dm.transOutY(yOut[:, None])[:, 0]
    
    iInd_indices = iInd[indices]
    jInd_indices = jInd[indices]
    
    df_data = {
        'obs' : obs, 
        'pred' : pred,
        'date_index' : iInd_indices, 
        'site_index' : jInd_indices
    }
    return pd.DataFrame(df_data)


def main(args):   
    dates = args.dates     
    hyperparams_path = args.hyperparams_path
    weights_path = args.weights_path
    fold = args.fold
    save_dir = args.save_dir

    with open(hyperparams_path, 'r') as f:
        hyperparams = json.loads(f.read()) # NOTE: why do we need f.read()?
    dataset = hyperparams['dataset']
    nh = hyperparams['nh'] # req for model arch
    rho = hyperparams['rho'] # req for model arch
    dropout = hyperparams['dropout']
    batch_size = hyperparams['batch_size']
    split_version = hyperparams['split_version']
    run_name = hyperparams['run_name']
    seed = hyperparams['seed']

    if dates == 'all':
        data = prepare_data_all(dataset, rho)
    elif dates == 'avail_insitu':
        data = prepare_data(dataset, rho)
    # xc = data['xc'] # TODO: do we need?

    # Load previously generated dataset splits
    splits_path = os.path.join(kPath.dirVeg, 'model', 'attention', split_version, 'subset.json')
    with open(splits_path) as f:
        splits_dict = json.load(f)
    split_indicies = {
        'train' : splits_dict[f'trainInd_k{fold}5'],
        'test_quality_sites' : splits_dict[f'testInd_k{fold}5'],
        'test_poor_sites' : splits_dict['testInd_underThresh']
    }

    # Set up model 
    xS, xM, _, _, _, _ = randomSubset(data, split_indicies['train'], batch_size) # Get sample to get shapes for model
    nTup = (xS.shape[-1], xM.shape[-1]) # (list[int]) Number of input features for each remote sensing source (i.e. Sentinel, Modis) 
    # lTup = (xS.shape[1], xM.shape[1])# TODO: do we need? # (tuple[int]) number of sampled days for each remote sensing source
    nxc = data['xc'].shape[-1] # (int): number of constant variables
    
    model = FinalModel(nTup, nxc, nh, dropout)
    model.load_state_dict(torch.load(weights_path))

    if dates == 'avail_insitu':
        train_df = get_predictions(data, split_indicies['train'], model)
        test_qual_df = get_predictions(data, split_indicies['test_quality_sites'], model)
        test_poor_df = get_predictions(data, split_indicies['test_poor_sites'], model)
        df = pd.concat([train_df, test_qual_df, test_poor_df])
        df['split'] = ['train'] * len(train_df) + ['test_qual'] * len(test_qual_df) + ['test_poor'] * len(test_poor_df)

        # convert site index to site ids
        sites_path = os.path.join(kPath.dirVeg, "model/data/singleDaily-nadgrid/info.csv")
        sites_df = pd.read_csv(sites_path)
        sites_dict = {
            "site" : sites_df.siteId,
            "site_index" : range(len(sites_df))
        }
        sites_df = pd.DataFrame(sites_dict)
        df = df.merge(sites_df)
        df = df.drop(["site_index"], axis=1)

        # convert elapsed days to dates
        date_range = pd.date_range(start='2016-10-15', end='2021-12-15')
        dates_dict ={
            "date" : date_range,
            "date_index" : range(len(date_range))
        }
        dates_df = pd.DataFrame(dates_dict)
        df = df.merge(dates_df)
        df = df.drop(["date_index"], axis=1)
        df.set_index('date', inplace=True)

        df['name'] = [run_name + f'_f{fold}'] * len(df)
        df['dataset'] = [dataset] * len(df)
        df['split_version'] = [split_version] * len(df)
        df['fold'] = [fold] * len(df)
        df['seed'] =  [seed] * len(df)
        save_path = os.path.join(save_dir, run_name + f'_f{fold}.csv')
        df.to_csv(save_path)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use pre-existing model for inference.')
    # admin
    parser.add_argument('--device', type=int, default=-1)
    # data
    parser.add_argument('--dates', type=str, choices=['all', 'avail_insitu'])
    parser.add_argument('--fold', type=int, choices=[0, 1, 2, 3, 4])
    # model
    parser.add_argument('--hyperparams_path', type=str)
    parser.add_argument('--weights_path', type=str)
    # save
    parser.add_argument('--save_dir', type=str, default=kPath.dirVeg)
    args = parser.parse_args()

    main(args)


