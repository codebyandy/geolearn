import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot
from sklearn.linear_model import LinearRegression
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
import torch
import os
import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

startDate = pd.datetime(1979, 1, 1)
endDate = pd.datetime(2020, 1, 1)
sn = 1
codeLst = usgs.newC
varX = gridMET.varLst

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']
t0 = time.time()

dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'Linear-W', 'B20Q')
dirOut = os.path.join(dirRoot, 'output')
dirPar = os.path.join(dirRoot, 'params')
for folder in [dirRoot, dirOut, dirPar]:
    if not os.path.exists(folder):
        os.mkdir(folder)


for kk, siteNo in enumerate(siteNoLst):
    print('{}/{} {:.2f}'.format(
        kk, len(siteNoLst), time.time()-t0))
    saveName = os.path.join(dirOut, siteNo)
    # if os.path.exists(saveName):
    #     continue
    varQ = '00060'
    varLst = codeLst+[varQ]+varX
    df = waterQuality.readSiteTS(siteNo, varLst=varLst, freq='W')

    dfX = pd.DataFrame({'date': df.index}).set_index('date')
    dfX = dfX.join(np.log(df[varQ]+sn)).rename(
        columns={varQ: 'logQ'})
    dfX = dfX.join(df[varX])
    yr = dfX.index.year.values
    t = yr+dfX.index.dayofyear.values/365
    # dfX['sinT'] = np.sin(2*np.pi*t)
    # dfX['cosT'] = np.cos(2*np.pi*t)
    # ind = np.where(yr < 2010)[0]
    ind = np.where(yr < 2020)[0]
    dfYP = pd.DataFrame(index=df.index, columns=codeLst)
    dfYP.index.name = 'date'
    dfXN = (dfX-dfX.min())/(dfX.max()-dfX.min())
    # dfXN = dfX
    for code in codeLst:
        x = dfXN.iloc[ind].values
        # y = np.log(df.iloc[ind][code].values+sn)
        y = df.iloc[ind][code].values
        [xx, yy], iv = utils.rmNan([x, y])
        if len(yy) > 0:
            # yy, ind = utils.rmExt(yv, p=2.5, returnInd=True)
            # xx = xv[ind, :]
            lrModel = LinearRegression()
            lrModel = lrModel.fit(xx, yy)
            b = dfXN.isna().any(axis=1)
            yp = lrModel.predict(dfXN[~b].values)
            dfYP.at[dfYP[~b].index, code] = yp
    dfYP.to_csv(saveName)

res = dfYP[code].values-df[code].values
fig, ax = plt.subplots(1, 1)
ax.plot(res, df['tmmn'], '*')
fig.show()
