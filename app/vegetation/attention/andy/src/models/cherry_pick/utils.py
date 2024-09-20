"""
This file contains some important methods and constants.
"""

import random
import numpy as np
import torch
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # for gpu 
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"seed {seed} set")

# satellite variable names
varS = ['VV', 'VH', 'vh_vv']
varL = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ndvi', 'ndwi', 'nirv']
varM = ["MCD43A4_b{}".format(x) for x in range(1, 8)]

# Days per satellite
bS = 8
bL = 6
bM = 10






# metrics = [
#         "poor_anomaly_coefdet",
#         "poor_anomaly_corrcoef",
#         "poor_anomaly_rmse",
#         "poor_obs_coefdet",
#         "poor_obs_corrcoef",
#         "poor_obs_rmse",
#         "poor_site_coefdet",
#         "poor_site_corrcoef",
#         "poor_site_rmse",
#         "qual_anomaly_coefdet",
#         "qual_anomaly_corrcoef",
#         "qual_anomaly_rmse",
#         "qual_obs_coefdet",
#         "qual_obs_corrcoef",
#         "qual_obs_rmse",
#         "qual_site_coefdet",
#         "qual_site_corrcoef",
#         "qual_site_rmse",
#         "test_minibatch_rmse",
#         "test_minibatch_coefdet",
#         "test_minibatch_loss",
#         "test_minibatch_coefcorr",
#         "train_minibatch_rmse",
#         "train_minibatch_coefdet",
#         "train_minibatch_loss",
#         "train_minibatch_coefcorr"
#     ]

# Define each metric to have its summary as "max"
# for metric in metrics:
#     wandb.define_metric(metric, summary="max")

        # Get metrics on subset of test set at the end of every epoch
        # model.eval()
        # with torch.no_grad():
        #     xS, xM, pS, pM, xcT, yT = randomSubset(data, split_indicies["train"] , split_indicies["test_quality"], 'test', batch_size)
        #     xTup, pTup = (), ()
        #     if satellites == "no_landsat":
        #         xTup = (xS, xM)
        #         pTup = (pS, pM)
        #     # else:
        #     #     xTup = (xS, xL, xM)
        #     #     pTup = (pS, pL, pM)  
        #     yP = model(xTup, pTup, xcT, lTup)
        #     loss = loss_fn(yP, yT)
        
        #     obs, pred = yP.detasch().numpy(), yT.detach().numpy()
        #     rmse = np.sqrt(np.mean((obs - pred) ** 2))
        #     corrcoef = np.corrcoef(obs, pred)[0, 1]
        #     coef_det = r2_score(obs, pred)

        #     metrics["test_minibatch_loss"] = loss_fn(yP, yT).item()
        #     metrics["test_minibatch_rmse"]= rmse
        #     metrics["test_minibatch_corrcoef"] = corrcoef
        #     metrics["test_minibatch_coefdet"] = coef_det

              # old_reported_model_path = os.path.join(saveFolder, f'model_ep{int(reported_metrics.epoch)}.pth')
        # new_reported_model_path = os.path.join(saveFolder, 'best_model.pth')
        # shutil.copyfile(old_reported_model_path, new_reported_model_path)




            # TODO: decide how to approach this (split wise)
    # quality_test_sites = dictSubset['testSite_k05']
    # poor_test_sites = dictSubset['testSite_underThresh']
    
    # quality_indices = np.where(np.isin(jInd, quality_test_sites))[0]
    # poor_indices = np.where(np.isin(jInd, poor_test_sites))[0]


    # # pred = get_metrics(data, indices, config)
    # jInd_ind = jInd[indices]
    # iInd_ind = iInd[indices]

    # pred = np.array(pred)
    # jInd_ind = np.array(jInd_ind)
    # iInd_ind = np.array(iInd_ind)

    # # TODO: which is train and which is test?
    # # TODO: output?
    # res = np.column_stack([pred, jInd_ind, iInd_ind])
    # path = os.path.join(kPath.dirVeg, 'transformer_lfmc_daily.npy')
    # np.save(path, res)