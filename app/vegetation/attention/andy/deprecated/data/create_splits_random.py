
from hydroDL import kPath
from hydroDL.data import dbVeg, DataModel
from hydroDL.master import basinFull, slurm, dataTs2Range

import numpy as np

import pandas as pd
import pickle
import os

import pdb

def prepare_dataset(dataName, rho):
    df = dbVeg.DataFrameVeg(dataName)
    dm = DataModel(X=df.x, XC=df.xc, Y=df.y)
    siteIdLst = df.siteIdLst
    dm.trans(mtdDefault='minmax')
    dataTup = dm.getData()
    pdb.set_trace()
    dataEnd, (iInd, jInd) = dataTs2Range(dataTup, rho, returnInd=True) # ind row/cols
    x, xc, y, yc = dataEnd

    # get indices of desired variables 
    varS = ['VV', 'VH', 'vh_vv']
    varL = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ndvi', 'ndwi', 'nirv']
    varM = ['Fpar', 'Lai']
    iS = [df.varX.index(var) for var in varS]
    iL = [df.varX.index(var) for var in varL]
    iM = [df.varX.index(var) for var in varM]

    # for each datapoint, collect Sentinal/Landsat/Modis data
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

    # filter datapoints 
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

    jSite, count = np.unique(jInd, return_counts=True) # site/counts
    countAry = np.array([[x, y] for y, x in sorted(zip(count, jSite))]) # sort
    nRm = sum(countAry[:, 1] < 5) # sum counts if < 5
    indSiteAll = countAry[nRm:, 0].astype(int) # remove those w/ < 5
    dictSubset = dict()

    for k in range(5):
        siteTest = indSiteAll[k::5] # slicing with step 5
        siteTrain = np.setdiff1d(indSiteAll, siteTest)
        indTest = np.where(np.isin(jInd, siteTest))[0] # keep if jInd in siteTest
        indTrain = np.where(np.isin(jInd, siteTrain))[0] # keep if jInd in siteTrain
        dictSubset['testSite_k{}5'.format(k)] = siteTest.tolist()
        dictSubset['trainSite_k{}5'.format(k)] = siteTrain.tolist()
        dictSubset['testInd_k{}5'.format(k)] = indTest.tolist()
        dictSubset['trainInd_k{}5'.format(k)] = indTrain.tolist()

    tInd = iInd
    siteInd = jInd
    indTrain = dictSubset['trainInd_k05']
    indTest = dictSubset['testInd_k05']

    data_tuple = (df, indTrain, indTest, nMat, pSLst, pLLst, pMLst, x, xc, yc) 

    # TODO: REMOVE
    return
    with open(os.path.join(kPath.dirVeg, 'data_tuple_random.pkl'), 'wb') as f:  # open a text file
        pickle.dump(data_tuple, f) # serialize the list

prepare_dataset("singleDaily", 45)