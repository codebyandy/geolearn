
from hydroDL import kPath
from hydroDL.data import dbVeg, DataModel
from hydroDL.master import basinFull, slurm, dataTs2Range

import numpy as np

import pandas as pd
import pickle
import os

import pdb

def prepare_dataset(dataName, rho):
    pdb.set_trace()
    df = dbVeg.DataFrameVeg(dataName)
    dm = DataModel(X=df.x, XC=df.xc, Y=df.y)
    siteIdLst = df.siteIdLst
    dm.trans(mtdDefault='minmax')
    dataTup = dm.getData()
    dataEnd, (iInd, jInd) = dataTs2Range(dataTup, rho, returnInd=True)
    x, xc, y, yc = dataEnd

    # get indinces of desired variables 
    varS = ['VV', 'VH', 'vh_vv']
    varL = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ndvi', 'ndwi', 'nirv']
    varM = ['Fpar', 'Lai']
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
    
    jSite, count = np.unique(jInd, return_counts=True)
    countAry = np.array([[x, y] for y, x in sorted(zip(count, jSite))])
    nRm = sum(countAry[:, 1] < 5)
    indSiteAll = countAry[nRm:, 0].astype(int)
    dictSubset = dict()

    # create splits
    lcIdx = [df.varXC.index(f"lc{i}") for i in range(1, 10)]
    lc_pct = df.xc[:, lcIdx]
    top_lcs = np.argmax(lc_pct, axis=1)
    site_lcs = pd.DataFrame({"lc": top_lcs})
    site_lcs = site_lcs[site_lcs.index.isin(indSiteAll)]
    
    size = 0
    num_folds = 3
    folds = {i : [] for i in range(num_folds)}
    for lc in np.unique(top_lcs):
        lc_subset = np.array(site_lcs[site_lcs.lc == lc].index)
        lc_subset = np.random.permutation(lc_subset)
        size += lc_subset.shape[0]
        split_size = round(len(lc_subset) / num_folds)
        
        i = 0
        for rand_fold in np.random.permutation(range(num_folds)):
            split_stop = len(lc_subset) if i == num_folds - 1 else split_size * (i + 1)
            folds[rand_fold].append(lc_subset[split_size * i : split_stop])
            i += 1

    folds = {i : np.concatenate(idxs) for (i, idxs) in folds.items()}

    data_folds = {}
    for i in range(num_folds):
        siteTest = folds[i]
        folds_for_train = []
        for j in range(num_folds):
            if j != i:
                folds_for_train.append(folds[j])
        siteTrain = np.concatenate(folds_for_train)
        indTrain = np.where(np.isin(jInd, siteTrain))[0]
        indTest = np.where(np.isin(jInd, siteTest))[0]
        data_tuple = (df, indTrain, indTest, nMat, pSLst, pLLst, pMLst, x, xc, yc) 
        data_folds[i] = data_tuple
    
    with open(os.path.join(kPath.dirVeg, 'TEST.pkl'), 'wb') as f:  # open a text file
        pickle.dump(data_folds, f) # serialize the list

prepare_dataset("singleDaily", 45)