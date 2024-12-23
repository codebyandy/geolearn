from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
from hydroDL.master import slurm
import importlib
import os
import json

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']

# comb model
dataName = 'rbDN5'
wqData = waterQuality.DataModelWQ(dataName)
codeLst = wqData.varC
labelLst = ['FP_QC', 'QFP_C', 'Q_C']
for label in labelLst:
    trainSet = 'comb-B10'
    outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
    basins.testModelSeq(outName, siteNoLst, wqData=wqData)
