import numpy as np
import scipy.stats

keyLst = ['Bias', 'RMSE', 'ubRMSE', 'Corr']


def statError(pred, target):
    ngrid, nt = pred.shape
    # Bias
    Bias = np.nanmean(pred - target, axis=1)
    # RMSE
    RMSE = np.sqrt(np.nanmean((pred - target)**2, axis=1))
    # ubRMSE
    predMean = np.tile(np.nanmean(pred, axis=1), (nt, 1)).transpose()
    targetMean = np.tile(np.nanmean(target, axis=1), (nt, 1)).transpose()
    predAnom = pred - predMean
    targetAnom = target - targetMean
    ubRMSE = np.sqrt(np.nanmean((predAnom - targetAnom)**2, axis=1))
    # rho R2 NSE
    Corr = np.full(ngrid, np.nan)
    R2 = np.full(ngrid, np.nan)
    NSE = np.full(ngrid, np.nan)
    PBiaslow = np.full(ngrid, np.nan)
    PBiashigh = np.full(ngrid, np.nan)
    for k in range(0, ngrid):
        x = pred[k, :]
        y = target[k, :]
        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
        if ind.shape[0] > 1:
            xx = x[ind]
            yy = y[ind]
            Corr[k] = scipy.stats.pearsonr(xx, yy)[0]
            yymean = yy.mean()
            SST = np.sum((yy-yymean)**2)
            SSReg = np.sum((xx-yymean)**2)
            SSRes = np.sum((yy-xx)**2)
            R2[k] = 1-SSRes/SST
            NSE[k] = 1-SSRes/SST
            # FHV the peak flows bias 2%
            # FLV the low flows bias bottom 30%, log space
            pred_sort = np.sort(xx)
            target_sort = np.sort(yy)
            indexlow = round(0.3*len(pred_sort))
            indexhigh = round(0.98*len(pred_sort))
            lowpred = pred_sort[:indexlow]
            highpred = pred_sort[indexhigh:]
            lowtarget = target_sort[:indexlow]
            hightarget = target_sort[indexhigh:]
            PBiaslow[k] = np.sum(lowpred - lowtarget)/np.sum(lowtarget)*100
            PBiashigh[k] = np.sum(highpred - hightarget)/np.sum(hightarget)*100
            outDict = dict(Bias=Bias, RMSE=RMSE, ubRMSE=ubRMSE, Corr=Corr, R2=R2, NSE=NSE, FLV=PBiaslow, FHV=PBiashigh)
    return outDict
