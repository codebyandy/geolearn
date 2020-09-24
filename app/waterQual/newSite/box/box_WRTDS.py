
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

codeLst = sorted(usgs.newC)
ep = 500
reTest = False
dataName = 'nbW'
wqData = waterQuality.DataModelWQ(dataName)
siteNoLst = wqData.info.siteNo.unique()
nSite = len(siteNoLst)
corrMat = np.full([nSite, len(codeLst), 4], np.nan)

dirWrtds = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS')
dfCorr = pd.read_csv(os.path.join(
    dirWrtds, '{}-{}-corr'.format('Y1', 'Y2')), index_col=0)
corrMat[:, :, 0] = dfCorr[codeLst].values
dirWrtds = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-F')
dfCorr = pd.read_csv(os.path.join(
    dirWrtds, '{}-{}-corr'.format('Y1', 'Y2')), index_col=0)
corrMat[:, :, 1] = dfCorr[codeLst].values

# single
labelLst = ['QFP_C']
for iLab, label in enumerate(labelLst):
    trainSet = 'comb-B16'
    testSet = 'comb-A16'
    outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
    master = basins.loadMaster(outName)
    yP, ycP = basins.testModel(
        outName, testSet, wqData=wqData, ep=ep, reTest=reTest)
    for iCode, code in enumerate(codeLst):
        ic = wqData.varC.index(code)
        ind = wqData.subset[testSet]
        info = wqData.info.iloc[ind].reset_index()
        ic = wqData.varC.index(code)
        if len(wqData.c.shape) == 3:
            p = yP[-1, :, master['varY'].index(code)]
            o = wqData.c[-1, ind, ic]
        elif len(wqData.c.shape) == 2:
            p = ycP[:, master['varYC'].index(code)]
            o = wqData.c[ind, ic]
        for iS, siteNo in enumerate(siteNoLst):
            indS = info[info['siteNo'] == siteNo].index.values
            rmse, corr = utils.stat.calErr(p[indS], o[indS])
            corrMat[iS, iCode, iLab+2] = corr

# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = ['WRTDS', 'LSTM', 'LSTM + Forcing']
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in [0, 2]:
        temp.append(corrMat[:, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst='bgr',
                      label2=labLst2, figsize=(12, 4), yRange=[0, 1])
# fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
#                       label2=labLst2, figsize=(12, 4), sharey=False)
fig.show()