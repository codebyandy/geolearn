import importlib
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
import json
import scipy
from astropy.timeseries import LombScargle
import matplotlib.gridspec as gridspec

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)

codeLst = sorted(usgs.newC)
ep = 500
reTest = False
dataName = 'rbWN5'
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)

# load all sequence
if False:
    dictLSTMLst = list()
    # LSTM
    labelLst = ['QTFP_C']
    for label in labelLst:
        dictLSTM = dict()
        trainSet = 'comb-B10'
        outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
        for k, siteNo in enumerate(siteNoLst):
            print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
            df = basins.loadSeq(outName, siteNo)
            dictLSTM[siteNo] = df
        dictLSTMLst.append(dictLSTM)
    # WRTDS
    dictWRTDS = dict()
    dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat',
                            'WRTDS-W', 'B10', 'output')
    for k, siteNo in enumerate(siteNoLst):
        print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
        saveFile = os.path.join(dirWRTDS, siteNo)
        df = pd.read_csv(saveFile, index_col=None).set_index('date')
        # df = utils.time.datePdf(df)
        dictWRTDS[siteNo] = df
    # Observation
    dictObs = dict()
    for k, siteNo in enumerate(siteNoLst):
        print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
        df = waterQuality.readSiteTS(
            siteNo, varLst=['00060']+codeLst, freq='W')
        dictObs[siteNo] = df

    # calculate correlation
    tt = np.datetime64('2010-01-01')
    t0 = np.datetime64('1980-01-01')
    ind1 = np.where((df.index.values < tt) & (df.index.values >= t0))[0]
    ind2 = np.where(df.index.values >= tt)[0]
    dictLSTM = dictLSTMLst[0]
    corrMat = np.full([len(siteNoLst), len(codeLst), 3], np.nan)
    for ic, code in enumerate(codeLst):
        for siteNo in dictSite[code]:
            indS = siteNoLst.index(siteNo)
            v1 = dictLSTM[siteNo][code].iloc[ind2].values
            v2 = dictWRTDS[siteNo][code].iloc[ind2].values
            v3 = dictObs[siteNo][code].iloc[ind2].values
            vv1, vv2, vv3 = utils.rmNan([v1, v2, v3], returnInd=False)
            rmse1, corr1 = utils.stat.calErr(vv1, vv2)
            rmse2, corr2 = utils.stat.calErr(vv1, vv3)
            rmse3, corr3 = utils.stat.calErr(vv2, vv3)
            corrMat[indS, ic, 0] = corr1
            corrMat[indS, ic, 1] = corr2
            corrMat[indS, ic, 2] = corr3

    # load basin attributes
    regionLst = ['ECO2_BAS_DOM', 'NUTR_BAS_DOM',
                 'HLR_BAS_DOM_100M', 'PNV_BAS_DOM']
    dfG = gageII.readData(siteNoLst=siteNoLst)
    fileT = os.path.join(gageII.dirTab, 'lookupPNV.csv')
    tabT = pd.read_csv(fileT).set_index('PNV_CODE')
    for code in range(1, 63):
        siteNoTemp = dfG[dfG['PNV_BAS_DOM'] == code].index
        dfG.at[siteNoTemp, 'PNV_BAS_DOM2'] = tabT.loc[code]['PNV_CLASS_CODE']
    dfG = gageII.updateCode(dfG)

# color mat
# cVar = 'CONTACT'
# cMat = dfG[cVar].values
# cMat = np.log(cMat+1)
# cR = [np.nanpercentile(cMat, 10), np.nanpercentile(cMat, 90)]
# cR = [np.nanmin(cMat), np.nanmax(cMat)]

# estimate travel time
d = dfG['ROCKDEPAVE'].values  # inches
a = dfG['DRAIN_SQKM'].values  # sqkm
c = dfG['AWCAVE'].values  # []
q = np.ndarray(len(siteNoLst))
for k, siteNo in enumerate(siteNoLst):
    q[k] = dictObs[siteNo]['00060'].mean()  # cubic feet / s
unitCov = 0.0254*10**6/0.3048**3/24/60/60/365  # year
tt = d*a*c/q * unitCov
cMat = np.log10(tt)
cVar = 'Estimated Travel time'
cR = [np.nanmin(cMat), np.nanmax(cMat)]
cR = [np.nanpercentile(cMat, 10), np.nanpercentile(cMat, 90)]

# plot 121
plt.close('all')
codeLst2 = ['00095', '00400', '00405', '00600', '00605',
            '00618', '00660', '00665', '00681', '00915',
            '00925', '00930', '00935', '00940', '00945',
            '00950', '00955', '70303', '71846', '80154']
nfy, nfx = [5, 4]

# codeLst2 = ['00095', '00618', '00915', '00925', '00935', '00955']
# nfy, nfx = [3, 2]

# attr vs diff
fig, axes = plt.subplots(nfy, nfx)
for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, nfy, nfx)
    ax = axes[j, i]
    ic = codeLst.index(code)
    x = cMat
    y = corrMat[:, ic, 1]**2-corrMat[:, ic, 2]**2
    ax.plot(x, y, '*')
    ax.plot([np.nanmin(x), np.nanmax(x)], [0, 0], 'k-')
    ax.set_ylim([-0.5, 0.5])
    # ax.set_xlim(cR)
    titleStr = '{} {} '.format(
        code, usgs.codePdf.loc[code]['shortName'])
    axplot.titleInner(ax, titleStr)
fig.show()


nyr = 5
ind1 = np.where(tt <= nyr)[0]
ind2 = np.where(tt > nyr)[0]
# significance test
dfS = pd.DataFrame(index=codeLst, columns=['corr1', 'corr2'])
for k, code in enumerate(codeLst):
    a = corrMat[ind1, k, 1]
    b = corrMat[ind1, k, 2]
    aa, bb = utils.rmNan([a, b], returnInd=False)
    # s, p = scipy.stats.ttest_ind(aa, bb)
    s, p = scipy.stats.wilcoxon(aa, bb)
    dfS.at[code, 'corr1'] = p
    a = corrMat[ind2, k, 1]
    b = corrMat[ind2, k, 2]
    aa, bb = utils.rmNan([a, b], returnInd=False)
    # s, p = scipy.stats.ttest_ind(aa, bb)
    s, p = scipy.stats.wilcoxon(aa, bb)
    dfS.at[code, 'corr2'] = p

# plot box corr
codeLst3 = ['00618', '00660', '00915', '00925', '70303']
labLst1 = list()
for code in codeLst3:
    lab = '{}\n{}'.format(usgs.codePdf.loc[code]['shortName'], code)
    labLst1.append(lab)
labLst2 = ['LSTM <= {} yr'.format(nyr), 'WRTDS < {} yr'.format(nyr),
           'LSTM > {} yr'.format(nyr), 'WRTDS > {} yr'.format(nyr)]
dataBox = list()
for code in codeLst3:
    temp = list()
    ic = codeLst.index(code)
    for i in [1, 2]:
        temp.append(corrMat[ind1, ic, i])
    for i in [1, 2]:
        temp.append(corrMat[ind2, ic, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst='mgrb',
                      label2=labLst2, figsize=(12, 4), yRange=[0, 1])
fig.show()
