from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt


outName = 'HBN-opt1'
trainSet = 'first80'
testSet = 'last20'

master = basins.loadMaster(outName)
wqData = waterQuality.DataModelWQ(master['dataName'])
p1, o1 = basins.testModel(outName, trainSet, wqData=wqData)
p2, o2 = basins.testModel(outName, testSet, wqData=wqData)

errMat1 = wqData.errBySite(p1, subset=trainSet)
errMat2 = wqData.errBySite(p2, subset=testSet)

# box plot


# plot
# get location
siteNoLst = wqData.info['siteNo'].unique().tolist()
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
codeSel = ['00955', '00940', '00915']
icLst = [wqData.varC.index(code) for code in codeSel]
codePdf = waterQuality.codePdf


def funcMap():
    figM, axM = plt.subplots(len(codeSel), 2, figsize=(8, 6))
    for k in range(len(codeSel)):
        ic = icLst[k]
        axplot.mapPoint(axM[k, 0], lat, lon, errMat2[:, ic, 0], s=6)
        axplot.mapPoint(axM[k, 1], lat, lon, errMat2[:, ic, 1], s=6)
    figP, axP = plt.subplots(len(codeSel), 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    info1 = wqData.extractSubsetInfo(trainSet)
    info2 = wqData.extractSubsetInfo(testSet)
    ind1 = info1[info1['siteNo'] == siteNo].index
    ind2 = info2[info2['siteNo'] == siteNo].index
    t1 = info1['date'][ind1].values.astype(np.datetime64)
    x1 = p1[ind1]
    y1 = o1[ind1]
    t2 = info2['date'][ind2].values.astype(np.datetime64)
    x2 = p2[ind2]
    y2 = o2[ind2]
    icLst = [wqData.varC.index(code) for code in codeSel]
    for k, ic in enumerate(icLst):
        axplot.plotTS(axP[k], t1, [x1[:, ic], y1[:, ic]], cLst='rb')
        axplot.plotTS(axP[k], t2, [x2[:, ic], y2[:, ic]], cLst='yg')


figplot.clickMap(funcMap, funcPoint)
