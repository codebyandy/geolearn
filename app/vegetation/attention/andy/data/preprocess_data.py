
# hydroDL module by Kuai Fang
from hydroDL.data import dbVeg
from hydroDL.data import DataModel
from hydroDL.master import dataTs2Range
from hydroDL import kPath

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import StratifiedKFold

import argparse
import json
import os


varS = ['VV', 'VH', 'vh_vv']
varM = ["MCD43A4_b{}".format(x) for x in range(1, 8)]

target_states = ['WASHINGTON',
                 'OREGON',
                 'IDAHO',
                 'MONTANA',
                 'WYOMING',
                 'CALIFORNIA',
                 'NEVADA',
                 'UTAH',
                 'COLORADO',
                 'ARIZONA',
                 'NEW MEXICO', 
                 'TEXAS'
                 ] 


def remove_obs_with_no_rs_data(df, x, xc, yc, iInd, jInd):
    iS = [df.varX.index(var) for var in varS]
    iM = [df.varX.index(var) for var in varM]

    # for each satellite, for each LFMC data point
    # create a list of days in the 91 day window that have data
    pSLst, pMLst = [], []
    nMat = np.zeros([yc.shape[0], 2]) # (num obs, num sources)
    for k in range(nMat.shape[0]):
        tempS = x[:, k, iS]  
        tempM = x[:, k, iM] 

        pS = np.where(~np.isnan(tempS).any(axis=1))[0]
        pM = np.where(~np.isnan(tempM).any(axis=1))[0]
        pSLst.append(pS)
        pMLst.append(pM)
        nMat[k, :] = [len(pS), len(pM)]

    num_obs = (nMat == 0).any(axis=1).sum()
    print(f"There are {num_obs} obsevations with no data for at least 1 source.")

    # keep data if there is at least 1 day of data for every source
    indKeep = np.where((nMat > 0).all(axis=1))[0]
    x = x[:, indKeep, :]
    xc = xc[indKeep, :]
    yc = yc[indKeep, :]
    nMat = nMat[indKeep, :]
    pSLst = [pSLst[k] for k in indKeep]
    pMLst = [pMLst[k] for k in indKeep]
    iInd = iInd[indKeep]
    jInd = jInd[indKeep] 

    return x, xc, yc, iInd, jInd


def remove_sites_not_in_western_us(df, countAry):
    # remove points not in target states
    lon_list = df.lon[countAry[:, 0]]
    lat_list = df.lat[countAry[:, 0]]
    points = [Point(lon, lat) for lon, lat in zip(lon_list, lat_list)]
    points_gdf = gpd.GeoDataFrame({'site': countAry[:, 0]}, geometry=points, crs='EPSG:4326')

    states_path = os.path.join(kPath.dirVeg, "shapefiles", "us", "States_shapefile.shp")
    states_gdf = gpd.read_file(states_path)
    states_gdf = states_gdf[['State_Code', 'State_Name', 'geometry']]

    target_states = ['WASHINGTON', 'OREGON', 'IDAHO', 'MONTANA', 'WYOMING', 'CALIFORNIA', 'NEVADA', 'UTAH', 'COLORADO', 'ARIZONA', 'NEW MEXICO', 'TEXAS'] 
    target_state_boundary = states_gdf[states_gdf['State_Name'].isin(target_states)]
    sites_within_states = np.array(gpd.sjoin(points_gdf, target_state_boundary).site)

    countAry = countAry[np.isin(countAry[:, 0], sites_within_states)]

    return countAry, sites_within_states


def split_sites_by_quality(countAry, site_obs_threshold):
    nRm = sum(countAry[:, 1] < site_obs_threshold)
    indSiteOverThresh = countAry[nRm:, 0].astype(int) 
    indSiteUnderThresh = countAry[:nRm, 0].astype(int) 
    indSiteOverThresh = np.sort(indSiteOverThresh)
    indSiteUnderThresh = np.sort(indSiteUnderThresh)

    return indSiteOverThresh, indSiteUnderThresh


def split_sites_by_land_cover(df, lc_pct, indSiteOverThresh, indSiteUnderThresh, jInd, land_cover_thresh, num_splits):
    # get land cover percentages for quality sites
    lc_idx = [df.varXC.index(var) for var in df.varXC[-6:]]
    lc_pct = df.xc[:, lc_idx]
    lc_pct = lc_pct[indSiteOverThresh]

    top_lc = np.argmax(lc_pct, axis=1)
    land_cover_array_over = np.stack((indSiteOverThresh, top_lc), axis=1)

    mixed_site_idxs = np.where(np.max(lc_pct, axis=1) < land_cover_thresh)
    land_cover_array_over[mixed_site_idxs, 1] = 5

    sites = land_cover_array_over[:, 0]
    land_cover_types = land_cover_array_over[:, 1]

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    
    dictSubset = {}
    for k, (train_index, test_index) in enumerate(skf.split(sites, land_cover_types)):
        train_sites, test_sites = sites[train_index], sites[test_index]
        dictSubset['testSite_k{}5'.format(k)] = test_sites.tolist()
        dictSubset['trainSite_k{}5'.format(k)] = train_sites.tolist()

        indTest = np.where(np.isin(jInd, test_sites))[0]
        indTrain = np.where(np.isin(jInd, train_sites))[0]
        dictSubset['testInd_k{}5'.format(k)] = indTest.tolist()
        dictSubset['trainInd_k{}5'.format(k)] = indTrain.tolist()

    dictSubset['testSite_underThresh'] = indSiteUnderThresh.tolist()
    dictSubset['testInd_underThresh'] = np.where(np.isin(jInd, indSiteUnderThresh))[0].tolist()

    return dictSubset



def preprocess(args):
    rho = args.rho
    data_name = args.data_name
    site_obs_threshold = args.site_obs_thresh
    land_cover_thresh = args.land_cover_thresh
    seed = args.seed
    num_splits = args.num_splits

    df = dbVeg.DataFrameVeg(data_name) 
    dm = DataModel(X=df.x, XC=df.xc, Y=df.y) 
    
    # normalization
    dm.trans(mtdDefault='minmax') 
    dataTup = dm.getData() 

    # prepare data for supervised learning, reshape to (num obs, rho, num varX) 
    dataEnd, (iInd, jInd) = dataTs2Range(dataTup, rho, returnInd=True) 
    x, xc, y, yc = dataEnd 
    
    iInd = np.array(iInd) # TODO: temporary fix
    jInd = np.array(jInd) # TODO: temporary fix

    print(f"There are {yc.shape[0]} total observations.")
    print(f"There are {df.y.shape[1]} total sites.")

    # filter out observations that have no data
    x, xc, yc, iInd, jInd = remove_obs_with_no_rs_data(df, x, xc, yc, iInd, jInd)

    jSite, count = np.unique(jInd, return_counts=True) # get the number of observations for each site
    countAry = np.array([[x, y] for y, x in sorted(zip(count, jSite))]) # (site, num obs)

    print(f"After removing obsevations that have no data for at least 1 satellite:")
    print(f"- There are {yc.shape[0]} total observations.")
    print(f"- There are {len(jSite)} sites.")

    # filter out sites that are not in the Western US
    countAry, sites_within_states = remove_sites_not_in_western_us(df, countAry)

    print(f"There are {len(jSite) - len(sites_within_states)} sites not in the Western US.")
    print("After removing sites not in the Western US:")
    print(f"- The are {np.isin(jInd, sites_within_states).sum()} observations.")
    print(f"- There are {countAry.shape[0]} sites.")
    
    # split sites by quality
    indSiteOverThresh, indSiteUnderThresh = split_sites_by_quality(countAry, site_obs_threshold)

    # land cover stratification
    dictSubset = split_sites_by_land_cover(df, df.xc, indSiteOverThresh, indSiteUnderThresh, jInd, land_cover_thresh, num_splits)

    # save dataset
    saveFolder = os.path.join(kPath.dirVeg, 'model', 'attention', f'new_stratified_s{seed}')
    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)

    subsetFile = os.path.join(saveFolder, 'subset.json')
    with open(subsetFile, 'w') as fp:
        json.dump(dictSubset, fp, indent=4)

    # #####
    # # FOR TESTING, REMOVE AFTERWARDS
    # ####
    split_version = 'stratified_s42'
    splits_path = os.path.join(kPath.dirVeg, 'model', 'attention', split_version, 'subset.json')
    with open(splits_path) as f:
        splits_dict = json.load(f)

    new_split_version = 'new_stratified_s42'
    new_splits_path = os.path.join(kPath.dirVeg, 'model', 'attention', new_split_version, 'subset.json')
    with open(new_splits_path) as f:
        new_splits_dict = json.load(f)

    assert splits_dict == new_splits_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--rho", type=int, default=45, help="rho.")
    parser.add_argument("--data_name", type=str, default='singleDaily-modisgrid-new-const', help="data_name.")
    parser.add_argument("--site_obs_thresh", type=int, default=30, help="Number of observations required for site to be considered quality.")
    parser.add_argument("--land_cover_thresh", type=float, default=0.4, help="Threshold for land cover percentage.")
    parser.add_argument("--num_splits", type=int, default=5, help="Number of splits for stratified KFold.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    preprocess(args)