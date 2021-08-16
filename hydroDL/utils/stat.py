import numpy as np
import warnings



def calErr(pred, obs, rmExt=False):
    a = obs
    b = pred
    if len(obs) == 0:
        return (np.nan, np.nan)
    if rmExt is True and len(a) != 0:
        aV = a[a < np.nanpercentile(a, 95)]
        aV = aV[aV > np.nanpercentile(a, 5)]
        ul = np.mean(aV)+np.std(aV)*5
        a[a > ul] = np.nan
    indV = np.where(~np.isnan(a) & ~np.isnan(b))
    if len(indV) > 0:
        rmse = np.sqrt(np.nanmean((a[indV]-b[indV])**2))
        corr = np.corrcoef(a[indV], b[indV])[0, 1]
        return (rmse, corr)
    else:
        return (np.nan, np.nan)


def calStat(pred, obs):
    if np.isnan(pred).all() or np.isnan(obs).all():
        return dict(Bias=np.nan, RMSE=np.nan, NSE=np.nan, Corr=np.nan)
    else:
        nash = calNash(pred, obs)
        rmse = calRmse(pred, obs)
        corr = calCorr(pred, obs)
        bias = calBias(pred, obs)
        outDict = dict(Bias=bias, RMSE=rmse, NSE=nash, Corr=corr)
        return outDict


def calNash(pred, obs):
    # data in [nT,nS]
    nash = 1-np.nansum((pred-obs)**2, axis=0) / \
        np.nansum((obs-np.nanmean(obs, axis=0))**2, axis=0)
    return nash


def calRmse(pred, obs):
    # data in [nT,nS]
    rmse = np.sqrt(np.nanmean((pred-obs)**2, axis=0))
    return rmse


def calCorr(pred, obs):
    mask = np.isnan(obs) | np.isnan(pred)
    A = pred.copy()
    B = obs.copy()
    A[mask] = np.nan
    B[mask] = np.nan
    mA = A - np.nanmean(A, axis=0)
    mB = B - np.nanmean(B, axis=0)
    p1 = np.nansum(mA*mB, axis=0)
    p2 = np.sqrt(np.nansum(mA**2, axis=0)*np.nansum(mB**2, axis=0))
    return p1/p2


def calCorrOld(pred, obs):
    # data in [nT,nS]
    bV = pred.ndim == 1
    if bV:
        pred = pred[:, None]
        obs = obs[:, None]
    [nT, nS] = pred.shape
    corr = np.full([nS], np.nan)
    for k in range(nS):
        a = pred[:, k]
        b = obs[:, k]
        indV = np.where(~np.isnan(a) & ~np.isnan(b))[0]
        with warnings.catch_warnings():
            if np.nanmin(b) == np.nanmax(b):
                print('WARNING connstant observation in calculating corrcoef')
                warnings.simplefilter('ignore', category=RuntimeWarning)
            corr[k] = np.corrcoef(a[indV], b[indV])[0, 1]
    if bV:
        return corr[0]
    else:
        return corr


def calBias(pred, obs):
    # data in [nT,nS]
    indV = np.where(~np.isnan(pred) & ~np.isnan(obs))
    bias = np.nanmean(pred[indV], axis=0)-np.nanmean(obs[indV], axis=0)
    return bias


def calPercent(x, p, rank=True):
    if rank:
        return np.nanpercentile(x, p*100)
    else:
        vmin = np.nanmin(x)
        vmax = np.nanmax(x)
        return (vmax-vmin)*p+vmin
