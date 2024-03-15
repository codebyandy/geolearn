import os
import socket
import collections


def initPathSMAP(dirDB, dirOut, dirResult):
    pathSMAP = collections.OrderedDict(
        DB_L3_Global=os.path.join(dirDB, 'Daily_L3'),
        DB_L3_NA=os.path.join(dirDB, 'Daily_L3_NA'),
        Out_L3_Global=os.path.join(dirOut, 'L3_Global'),
        Out_L3_NA=os.path.join(dirOut, 'L3_NA'),
        outTest=os.path.join(dirOut, 'Test'),
        dirDB=dirDB,
        dirOut=dirOut,
        dirResult=dirResult)
    return pathSMAP


hostName = socket.gethostname()

if hostName[:2] == 'sh':
    host = 'sherlock'
    dirJob = r'/scratch/users/avhuynh/jobs/'
    dirVeg=r'/home/users/avhuynh/lfmc/data/'
    dirCode = r'/home/users/avhuynh/lfmc/geolearn/app/vegetation/attention/'
elif hostName == 'mac':
    host =  'mac'
    # dirData = r'Documents/lfmc/geolearn/hydroDL/data'
    dirVeg = r'/Users/andyhuynh/Documents/lfmc/data/'
    dirCode = r'/Users/andyhuynh/Documents/lfmc/geolearn/app/vegetation/attention/'
    dirJob = r'/Users/andyhuynh/Documents/lfmc/data/jobs/'


