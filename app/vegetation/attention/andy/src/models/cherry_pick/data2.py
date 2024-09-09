"""
This file contains utility functions for preprocessing and sampling a batch of data.

Usage:
    - Use the `prepare_data` function to load and preprocess the data.
    - Use the `randomSubset` function to sample subsets of the data for model training and evaluation.
"""

# hydroDL module by Kuai Fang
import hydroDL.data.dbVeg
from hydroDL.data import dbVeg
from hydroDL.data import DataModel
from hydroDL.master import dataTs2Range
from hydroDL import kPath

import numpy as np
import torch

from utils import varS, varL, varM # Satellite variable names
from utils import bS, bL, bM # Days per satellite


def randomSubset(data, trainInd, testInd, opt='train', batch=1000):
    """
    Return a batch of remote sensing/constant variable and LFMC data.
    Randomly sample `batch` observations.
    Randomly sample (cherry-pick) available days for each observation.

    Args:
        data (tuple): data tuple returned by prepare_dataset method
        trainInd (list[int]): indices of train sites
        testInd (list[int]): indices of test sites
        opt (str): train or test
        batch (int): batch size 
    Returns:
        xS (torch.Tensor): Sampled Sentinel raw values, of shape (batch size x bS x number Sentinel inputs)
        xL (torch.Tensor): Sampled Landsat raw values, of shape (batch size x bL x number Landsat inputs)
        xM (torch.Tensor): Sampled MODIS raw values, of shape (batch size x bM x number MODIS inputs)
        pS (torch.Tensor): Normalized position values for Sentinel, of shape (batch size x bS)
        pL (torch.Tensor): Normalized position values for Landsat, of shape (batch size x bL)
        pM (torch.Tensor): Normalized position values for MODIS, of shape (batch size x bM)
        xc (torch.Tensor): Constant variables, of shape (batch size, number constant variables)
        yc (torch.Tensor): LFMC data, of shape (batch size, 1)
    """
    df, _, _, _, nMat, pSLst, pMLst, x, rho, xc, yc = data
    
    # Randomly sample `batch` number of observations.
    if opt == 'train':
        indSel = np.random.permutation(trainInd)[0:batch]
    else:
        indSel = testInd

    iS = [df.varX.index(var) for var in varS]
    # iL = [df.varX.index(var) for var in varL]
    iM = [df.varX.index(var) for var in varM]
    ns = len(indSel)
    
    # Randomly sample (cherry-pick) available days with replacement
    # import pdb
    # pdb.set_trace()
    rS = np.random.randint(0, nMat[indSel, 0], [bS, ns]).T
    # rL = np.random.randint(0, nMat[indSel, 1], [bL, ns]).T
    # rM = np.random.randint(0, nMat[indSel, 2], [bM, ns]).T
    rM = np.random.randint(0, nMat[indSel, 1], [bM, ns]).T
    pS = np.stack([pSLst[indSel[k]][rS[k, :]] for k in range(ns)], axis=0)
    # pL = np.stack([pLLst[indSel[k]][rL[k, :]] for k in range(ns)], axis=0)
    pM = np.stack([pMLst[indSel[k]][rM[k, :]] for k in range(ns)], axis=0)
    matS1 = x[:, indSel, :][:, :, iS]
    # matL1 = x[:, indSel, :][:, :, iL]
    matM1 = x[:, indSel, :][:, :, iM]
    xS = np.stack([matS1[pS[k, :], k, :] for k in range(ns)], axis=0)
    # xL = np.stack([matL1[pL[k, :], k, :] for k in range(ns)], axis=0)
    xM = np.stack([matM1[pM[k, :], k, :] for k in range(ns)], axis=0)
    
    pS = (pS - rho) / rho
    # pL = (pL - rho) / rho
    pM = (pM - rho) / rho
    
    return (
        torch.tensor(xS, dtype=torch.float32),
        # torch.tensor(xL, dtype=torch.float32),
        torch.tensor(xM, dtype=torch.float32),
        torch.tensor(pS, dtype=torch.float32),
        # torch.tensor(pL, dtype=torch.float32),
        torch.tensor(pM, dtype=torch.float32),
        torch.tensor(xc[indSel, :], dtype=torch.float32),
        torch.tensor(yc[indSel, 0], dtype=torch.float32),
    )


def prepare_data(dataName, rho):
    """
    Loads the data, normalizes it, and processes it into the required shape and format 
    for further analysis. It identifies the available days for each observation from various remote 
    sensing sources (Sentinel, Landsat, MODIS) and filters the observations to retain only those with 
    data available from all sources.

    Args:
        dataName (str): The name or path of the dataset to be loaded.
        rho (int): The time window size for each observation.

    Returns:
        tuple: A tuple containing the following elements:
            - df (DataFrameVeg): A custom DataFrameVeg object containing the loaded data.
            - dm (DataModel): A custom DataModel object containing the normalized data.
            - iInd (np.array): Array of day indices for the observations.
            - jInd (np.array): Array of site indices for the observations.
            - nMat (np.array): Matrix indicating the number of available days with data for each satellite 
                               (Sentinel, Landsat, MODIS) for each observation.
            - pSLst (list): List of arrays indicating the days with available data for Sentinel for each observation.
            - pLLst (list): List of arrays indicating the days with available data for Landsat for each observation.
            - pMLst (list): List of arrays indicating the days with available data for MODIS for each observation.
            - x (np.array): Array of raw values for each observation in the shape (rho, number of observations, number of input features).
            - rho (int): The time window size for each observation.
            - xc (np.array): Array of constant variables for the observations.
            - yc (np.array): Array of LFMC data for the observations.
    """
    # Load data with custom DataFrameVeg and DataModel classes
    df = dbVeg.DataFrameVeg(dataName)
    dm = DataModel(X=df.x, XC=df.xc, Y=df.y)
    dm.trans(mtdDefault='minmax') # Min-max normalization
    dataTup = dm.getData()

    # To convert data to shape (Number of observations, rho, number of input features)
    dataEnd, (iInd, jInd) = dataTs2Range(dataTup, rho, returnInd=True) # iInd: day, jInd: site
    x, xc, _, yc = dataEnd 
    # import pdb
    # pdb.set_trace()
   
    iInd = np.array(iInd) # TODO: Temporary fix
    jInd = np.array(jInd) # TODO: emporary fix
    
    iS = [df.varX.index(var) for var in varS]
    iL = [df.varX.index(var) for var in varL]
    iM = [df.varX.index(var) for var in varM]
    
    # For each remote sensing source (i.e. Sentinel, MODIS), for each LFMC observaton,
    # create a list of days in the rho-window that have data 
    # nMat: Number of days each satellite has data for, of shape (# obsevations, # satellites)
    # pSLst, pLLst, pMLst = list(), list(), list()
    # nMat = np.zeros([yc.shape[0], 3])
    # for k in range(nMat.shape[0]):
    #     tempS = x[:, k, iS]
    #     pS = np.where(~np.isnan(tempS).any(axis=1))[0]
    #     tempL = x[:, k, iL]
    #     pL = np.where(~np.isnan(tempL).any(axis=1))[0]
    #     tempM = x[:, k, iM]
    #     pM = np.where(~np.isnan(tempM).any(axis=1))[0]
    #     pSLst.append(pS)
    #     pLLst.append(pL)
    #     pMLst.append(pM)
    #     nMat[k, :] = [len(pS), len(pL), len(pM)]

    pSLst, pMLst = list(), list()
    nMat = np.zeros([yc.shape[0], 2])
    for k in range(nMat.shape[0]):
        tempS = x[:, k, iS]
        pS = np.where(~np.isnan(tempS).any(axis=1))[0]
        pSLst.append(pS)
        tempM = x[:, k, iM]
        pM = np.where(~np.isnan(tempM).any(axis=1))[0]
        pMLst.append(pM)
        nMat[k, :] = [len(pS), len(pM)]
    
    # only keep if data if there is at least 1 day of data for each remote sensing source
    # import pdb
    # pdb.set_trace()
    indKeep = np.where((nMat > 0).all(axis=1))[0]
    x = x[:, indKeep, :]
    xc = xc[indKeep, :]
    yc = yc[indKeep, :]
    nMat = nMat[indKeep, :]
    pSLst = [pSLst[k] for k in indKeep]
    # pLLst = [pLLst[k] for k in indKeep]
    pMLst = [pMLst[k] for k in indKeep]
    
    jInd = jInd[indKeep]
    iInd = iInd[indKeep]

    data = (df, dm, iInd, jInd, nMat, pSLst, pMLst, x, rho, xc, yc)
    return data

# """
# This file contains utility functions for preprocessing and sampling a batch of data.

# Usage:
#     - Use the `prepare_data` function to load and preprocess the data.
#     - Use the `randomSubset` function to sample subsets of the data for model training and evaluation.
# """

# # module by Kuai Fang
# from hydroDL.data import dbVeg
# from hydroDL.data import DataModel
# from hydroDL.master import dataTs2Range

# import numpy as np
# import torch

# from utils import varS, varL, varM # Satellite variable names
# from utils import bS, bL, bM # Days per satellite


# def randomSubset(data, indices, batch, opt='train'):
#     """
#     Return a batch of remote sensing/constant variable and LFMC data.
#     Randomly sample `batch` observations.
#     Randomly sample (cherry-pick) available days for each observation.

#     Args:
#         data (tuple): data tuple returned by prepare_dataset method
#         trainInd (list[int]): indices of train sites
#         testInd (list[int]): indices of test sites
#         opt (str): train or test
#         batch (int): batch size 
#     Returns:
#         xS (torch.Tensor): Sampled Sentinel raw values, of shape (batch size x bS x number Sentinel inputs)
#         xL (torch.Tensor): Sampled Landsat raw values, of shape (batch size x bL x number Landsat inputs)
#         xM (torch.Tensor): Sampled MODIS raw values, of shape (batch size x bM x number MODIS inputs)
#         pS (torch.Tensor): Normalized position values for Sentinel, of shape (batch size x bS)
#         pL (torch.Tensor): Normalized position values for Landsat, of shape (batch size x bL)
#         pM (torch.Tensor): Normalized position values for MODIS, of shape (batch size x bM)
#         xc (torch.Tensor): Constant variables, of shape (batch size, number constant variables)
#         yc (torch.Tensor): LFMC data, of shape (batch size, 1)
#     """
#     df = data['dataFrame'] 
#     nMat = data['sat_avail_per_obs']
#     pSLst = data['s_days_per_obs'] 
#     pMLst = data['m_days_per_obs']
#     rho = data['rho']
#     x = data['x']
#     xc = data['xc']
#     yc = data['yc']

#     # Randomly sample `batch` number of observations.
#     if opt == 'train':
#         indSel = np.random.permutation(indices)[0:batch]
#     else:
#         indSel = indices
#     iS = [df.varX.index(var) for var in varS]
#     iM = [df.varX.index(var) for var in varM]
#     ns = len(indSel)
    
#     # Randomly sample (cherry-pick) available days with replacement
#     rS = np.random.randint(0, nMat[indSel, 0], [bS, ns]).T
#     rM = np.random.randint(0, nMat[indSel, 1], [bM, ns]).T
#     pS = np.stack([pSLst[indSel[k]][rS[k, :]] for k in range(ns)], axis=0)
#     pM = np.stack([pMLst[indSel[k]][rM[k, :]] for k in range(ns)], axis=0)
#     matS1 = x[:, indSel, :][:, :, iS]
#     matM1 = x[:, indSel, :][:, :, iM]
#     xS = np.stack([matS1[pS[k, :], k, :] for k in range(ns)], axis=0)
#     xM = np.stack([matM1[pM[k, :], k, :] for k in range(ns)], axis=0)
    
#     pS = (pS - rho) / rho
#     pM = (pM - rho) / rho
    
#     return (
#         torch.tensor(xS, dtype=torch.float32),
#         torch.tensor(xM, dtype=torch.float32),
#         torch.tensor(pS, dtype=torch.float32),
#         torch.tensor(pM, dtype=torch.float32),
#         torch.tensor(xc[indSel, :], dtype=torch.float32),
#         torch.tensor(yc[indSel, 0], dtype=torch.float32),
#     )


# def prepare_data(dataName, rho):
#     """
#     Loads the data, normalizes it, and processes it into the required shape and format 
#     for further analysis. It identifies the available days for each observation from various remote 
#     sensing sources (Sentinel, Landsat, MODIS) and filters the observations to retain only those with 
#     data available from all sources.

#     Args:
#         dataName (str): The name or path of the dataset to be loaded.
#         rho (int): The time window size for each observation.

#     Returns:
#         tuple: A tuple containing the following elements:
#             - df (DataFrameVeg): A custom DataFrameVeg object containing the loaded data.
#             - dm (DataModel): A custom DataModel object containing the normalized data.
#             - iInd (np.array): Array of day indices for the observations.
#             - jInd (np.array): Array of site indices for the observations.
#             - nMat (np.array): Matrix indicating the number of available days with data for each satellite 
#                                (Sentinel, Landsat, MODIS) for each observation.
#             - pSLst (list): List of arrays indicating the days with available data for Sentinel for each observation.
#             - pLLst (list): List of arrays indicating the days with available data for Landsat for each observation.
#             - pMLst (list): List of arrays indicating the days with available data for MODIS for each observation.
#             - x (np.array): Array of raw values for each observation in the shape (rho, number of observations, number of input features).
#             - rho (int): The time window size for each observation.
#             - xc (np.array): Array of constant variables for the observations.
#             - yc (np.array): Array of LFMC data for the observations.
#     """
#     # Load data with custom DataFrameVeg and DataModel classes
#     df = dbVeg.DataFrameVeg(dataName)
#     dm = DataModel(X=df.x, XC=df.xc, Y=df.y)
#     dm.trans(mtdDefault='minmax') # Min-max normalization
#     dataTup = dm.getData()

#     # To convert data to shape (Number of observations, rho, number of input features)
#     dataEnd, (iInd, jInd) = dataTs2Range(dataTup, rho, returnInd=True) # iInd: day, jInd: site
#     x, xc, _, yc = dataEnd 
   
#     iInd = np.array(iInd) # TODO: Temporary fix
#     jInd = np.array(jInd) # TODO: emporary fix
    
#     iS = [df.varX.index(var) for var in varS]
#     iM = [df.varX.index(var) for var in varM]
    
#     # For each remote sensing source (i.e. Sentinel, MODIS), for each LFMC observaton,
#     # create a list of days in the rho-window that have data 
#     # nMat: Number of days each satellite has data for, of shape (# obsevations, # satellites)
#     pSLst, pMLst = list(), list()
#     nMat = np.zeros([yc.shape[0], 2])
#     for k in range(nMat.shape[0]):
#         tempS = x[:, k, iS]
#         pS = np.where(~np.isnan(tempS).any(axis=1))[0]
#         pSLst.append(pS)
#         tempM = x[:, k, iM]
#         pM = np.where(~np.isnan(tempM).any(axis=1))[0]
#         pMLst.append(pM)
#         nMat[k, :] = [len(pS), len(pM)]
    
#     # only keep if data if there is at least 1 day of data for each remote sensing source
#     indKeep = np.where((nMat > 0).all(axis=1))[0]
#     x = x[:, indKeep, :]
#     xc = xc[indKeep, :]
#     yc = yc[indKeep, :]
#     nMat = nMat[indKeep, :]
#     pSLst = [pSLst[k] for k in indKeep]
#     pMLst = [pMLst[k] for k in indKeep]
    
#     jInd = jInd[indKeep]
#     iInd = iInd[indKeep]

#     data = {
#         'dataFrame' : df,
#         'dataModel' : dm,
#         'day_indices' : iInd,
#         'site_indices' : jInd,
#         'sat_avail_per_obs' : nMat, 
#         's_days_per_obs' : pSLst,
#         'm_days_per_obs': pMLst,
#         'rho' : rho,
#         'x' : x,
#         'xc' : xc,
#         'yc' : yc
#     }

#     return data