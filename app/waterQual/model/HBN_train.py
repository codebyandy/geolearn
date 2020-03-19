from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII
from hydroDL.master import basins

import pandas as pd
import numpy as np
import os
import time

# all gages
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
dfHBN = pd.read_csv(os.path.join(kPath.dirData, 'USGS', 'inventory', 'HBN.csv'), dtype={
    'siteNo': str}).set_index('siteNo')
siteNoHBN = [siteNo for siteNo in dfHBN.index.tolist()
             if siteNo in siteNoLstAll]

# wrap up data
if not waterQuality.exist('HBN'):
    wqData = waterQuality.DataModelWQ.new('HBN', siteNoHBN)
wqData = waterQuality.DataModelWQ('HBN')
ind = wqData.subset['first80']
indRm = wqData.indByComb(['00010', '00095'])
indTrain = np.setdiff1d(ind, indRm)
wqData.saveSubset('first80-rm2', indTrain)


for opt in [1, 2, 3, 4]:
    for trainName, trainStr in zip(['first80', 'first80-rm2'], ['', '-rm2']):
        saveName = 'HBN-opt'+str(opt)+trainStr
        caseName = basins.wrapMaster('HBN', trainName, batchSize=[
                                     None, 200], optQ=opt, saveName=saveName)


# if not waterQuality.exist('HBN-30d'):
#     wqData = waterQuality.DataModelWQ.new('HBN-30d', siteNoHBN, rho=30)
# if not waterQuality.exist('HBN-5s'):
#     wqData = waterQuality.DataModelWQ.new('HBN-5s', siteNoHBN[:5])
# if not waterQuality.exist('HBN-5s-30d'):
#     wqData = waterQuality.DataModelWQ.new('HBN-5s-30d', siteNoHBN[:5], rho=30)

# nE = 100
# sE = 50
# caseName = basins.wrapMaster('HBN-5s', 'first80', saveEpoch=sE, nEpoch=nE,
#                              batchSize=[None, 200], optQ=1, saveName='HBN-5s-opt1')
# basins.trainModelTS(caseName)
# caseName = basins.wrapMaster('HBN-5s-30d', 'first80', saveEpoch=sE, nEpoch=nE,
#                              batchSize=[None, 200], optQ=1, saveName='HBN-5s-30d-opt1')
# basins.trainModelTS(caseName)