from hydroDL.master import slurm

path = "/Users/andyhuynh/Documents/lfmc/geolearn/app/vegetation/attention/"

jobName="test.sh"
cmdLine=f"python3 {path}data.py" 

slurm.submitJobGPU(jobName, cmdLine)