from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL.master import slurm

wqData = waterQuality.DataModelWQ('basinRef')
ind1 = wqData.indByRatio(0.8)
ind2 = wqData.indByRatio(0.8, first=False)
wqData.saveSubset(['first80', 'last20'], [ind1, ind2])

caseName = basins.wrapMaster(
    'basinRef', 'first80', batchSize=[None, 500], optQ=1, outName='basinRef-opt1')
cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M'
slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=24)

caseName = basins.wrapMaster(
    'basinRef', 'first80', batchSize=[None, 500], optQ=1, outName='basinRef-opt1')
cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M'
slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=24)
