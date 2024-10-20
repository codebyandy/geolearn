from hydroDL.data import gageII
from hydroDL.master import slurm
from hydroDL.data import dbBasin
from hydroDL.master import basinFull

dataName = 'N200'
label = 'QFPRT2C'
# DF = dbBasin.DataFrameBasin(dataName)
rho = 365
nbatch = 500
hs = 256
trainSet = 'rmYr5'
testSet = 'pkYr5'


varX = dbBasin.label2var(label.split('2')[0])
mtdX = dbBasin.io.extractVarMtd(varX)
varY = dbBasin.label2var(label.split('2')[1])
mtdY = dbBasin.io.extractVarMtd(varY)
varXC = gageII.varLst
mtdXC = dbBasin.io.extractVarMtd(varXC)
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)
# outName = 'test-{}-{}-{}'.format(dataName, label, trainSet)
outName = 'test1'
dictP = basinFull.wrapMaster(outName=outName, dataName=dataName, trainSet=trainSet,
                             nEpoch=500, saveEpoch=50, optBatch='Weight',
                             varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                             mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC,
                             hiddenSize=hs, batchSize=[rho, nbatch],
                             nIterEp=50, crit='RmseLoss3D')
cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
slurm.submitJobGPU(outName, cmdP.format(outName), nH=4, nM=64)
# basinFull.trainModel(outName)

outName = 'test2'
dictP = basinFull.wrapMaster(outName=outName, dataName=dataName, trainSet=trainSet,
                             nEpoch=500, saveEpoch=50, optBatch='Weight',
                             varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                             mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC,
                             hiddenSize=hs, batchSize=[rho, nbatch],
                             nIterEp=50, crit='RmseLoss')
cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
slurm.submitJobGPU(outName, cmdP.format(outName), nH=4, nM=64)
