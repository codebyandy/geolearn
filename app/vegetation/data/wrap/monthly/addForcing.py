import numpy as np
import netCDF4
from hydroDL import kPath
import os
import pandas as pd
import hydroDL.data.cmip.io
import hydroDL.data.gridMET.io
import importlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from hydroDL.post import mapplot, axplot, figplot
import time

varLst = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']
yrLst = list(range(2016, 2022))
    # read site
siteFile = os.path.join(kPath.dirVeg, 'model', 'data', 'site.csv')
dfSite = pd.read_csv(siteFile)

# read forcing
sdN = '2016-08-31'
edN = '2021-12-15'
tN = pd.date_range(sdN, edN, freq='SM')
matF = np.full([len(tN), len(dfSite), len(varLst)], np.nan)
for k,var in enumerate(varLst):
    dfM = pd.read_csv(os.path.join(kPath.dirVeg, 'forcings', '{}.csv'.format(var)), index_col=0)
    dfM.index=pd.to_datetime(dfM.index)
    tM = dfM.index.values
    _, indT1, indT2 = np.intersect1d(tM, tN, return_indices=True)

    v = dfM[dfSite['siteId']].values[indT1, :]
    matF[:, :, k] = v



# append forcings
outFile = os.path.join(kPath.dirVeg, 'model', 'data', 'trainData.npz')
# load data
data = np.load(outFile)
x = data['x']
y = data['y']
xc = data['xc']
t = data['t']
varX = data['varX']
varY = data['varY']
varXC = data['varXC']

x = np.concatenate([x, matF], axis=-1)
varX = list(varX) + varLst

np.savez(outFile, varY=varY, y=y, varX=varX, x=x, t=t, varXC=varXC, xc=xc)
