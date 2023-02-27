from hydroDL.data import dbBasin, usgs, gageII, gridMET, GLASS
import os
from hydroDL import kPath
import numpy as np
import pandas as pd
import json

saveName = 'siteNoLst_79_23'
saveFile = os.path.join(kPath.dirUsgs, 'siteSel', saveName)
with open(saveFile, 'r') as f:
    dictSite = json.load(f)

label = 'B200'

# DF for single codes
for code in ['00915', '00955', '00618']:
    siteName = '{}-{}'.format(code, label)
    dataName = '{}-{}'.format(code, label)
    siteNoLst = dictSite[siteName]
    DF = dbBasin.DataFrameBasin.new(
        dataName,
        siteNoLst,
        varC=[code],
        varQ=usgs.varQ,
        varF=gridMET.varLst,
        varG=gageII.varLstEx,
        sdStr='1979-01-01',
        edStr='2023-01-01',
    )

# subset
# pick by year
for code in ['00915', '00955', '00618']:
    dataName = '{}-{}'.format(code, label)
    DF = dbBasin.DataFrameBasin(dataName)
    sy = DF.t[0].astype(object).year
    ey = DF.t[-1].astype(object).year
    yrAry = np.array(range(sy, ey))
    for k in range(5):
        yrIn = yrAry[yrAry % 5 == k]
        t1 = dbBasin.func.pickByYear(DF.t, yrIn)
        t2 = dbBasin.func.pickByYear(DF.t, yrIn, pick=False)
        DF.createSubset('pkYr5b{}'.format(k), dateLst=t1)
        DF.createSubset('rmYr5b{}'.format(k), dateLst=t2)

    # before after 2015
    DF.saveSubset('B15', ed='2015-12-31')
    DF.saveSubset('A15', sd='2016-01-01')

