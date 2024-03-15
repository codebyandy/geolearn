from hydroDL.master import slurm
from hydroDL import kPath

jobName="test.sh"
cmdLine=f"python3 {kPath.dirCode}data.py" 

slurm.submitJobGPU(jobName, cmdLine)