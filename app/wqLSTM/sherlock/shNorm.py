from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS
from hydroDL.master import slurm
from hydroDL.data import dbBasin
from hydroDL.master import basinFull

dataNameLst = ['G400Norm', 'G200Norm']

for dataName in dataNameLst:
    # DF = dbBasin.DataFrameBasin(dataName)
    varX = gridMET.varLst+ntn.varLst+GLASS.varLst+dbBasin.io.varTLst
    mtdX = dbBasin.io.extractVarMtd(varX)
    varY = [c+'-N' for c in usgs.newC]
    mtdY = dbBasin.io.extractVarMtd(varY)
    varXC = gageII.varLst + \
        [c+'-M' for c in usgs.newC] + [c+'-S' for c in usgs.newC]
    mtdXC = dbBasin.io.extractVarMtd(varXC)
    varYC = None
    mtdYC = dbBasin.io.extractVarMtd(varYC)
    sd = '1982-01-01'
    ed = '2009-12-31'
    outName = dataName
    trainSet = 'rmRT20'
    testSet = 'pkRT20'
    dictP = basinFull.wrapMaster(outName=outName, dataName=dataName, trainSet=trainSet,
                                 nEpoch=2000, batchSize=[365, 500], nIterEp=100,
                                 varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                                 mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC)
    cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
    slurm.submitJobGPU(outName, cmdP.format(outName), nH=24, nM=64)
