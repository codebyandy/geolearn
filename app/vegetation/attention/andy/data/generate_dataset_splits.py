# hydroDL module by Kuai Fang
from hydroDL.data import dbVeg
from hydroDL.data import DataModel
from hydroDL.master import dataTs2Range
from hydroDL import kPath

import numpy as np
import json
import os


def generate_splits(rho, d):
    rho = 45 # init rho
    dataName = 'singleDaily-modisgrid-new-const' # init dataName
    df = dbVeg.DataFrameVeg(dataName) # create DataFrameVeg class 
    dm = DataModel(X=df.x, XC=df.xc, Y=df.y) # (?) create DataModel class (contains many confusing functions) 
    siteIdLst = df.siteIdLst # get site list
    dm.trans(mtdDefault='minmax') # (?) some sort of data normalization
    dataTup = dm.getData() # get x, xc, y, and yc
    dataEnd, (iInd, jInd) = dataTs2Range(dataTup, rho, returnInd=True) # get data into form (# LFMC, 91 day window, varX) 
    x, xc, y, yc = dataEnd # data from dataTs2Range
    iInd = np.array(iInd)
    jInd = np.array(jInd)

    # get indices of variables of interest
    varS = ['VV', 'VH', 'vh_vv']
    varM = ["MCD43A4_b{}".format(x) for x in range(1, 8)]
    iS = [df.varX.index(var) for var in varS]
    iM = [df.varX.index(var) for var in varM]

    # for each satellite, for each LFMC data point
    # create a list of days in the 91 day window that have data

    # nMat -- (# obs, # satellites)
    # nMat contains # of days each satellite has data for
    pSLst, pMLst = list(), list()
    ns = yc.shape[0]
    nMat = np.zeros([yc.shape[0], 2])
    for k in range(nMat.shape[0]):
        tempS = x[:, k, iS] # x (rho, LFMC, varX) 
        pS = np.where(~np.isnan(tempS).any(axis=1))[0]
        tempM = x[:, k, iM] # x (rho, LFMC, varX) 
        pM = np.where(~np.isnan(tempM).any(axis=1))[0]
        pSLst.append(pS)
        pMLst.append(pM)
        nMat[k, :] = [len(pS), len(pM)]

    # only keep data if there is at least 1 day of data for 
    # each satellite
    indKeep = np.where((nMat > 0).all(axis=1))[0]
    x = x[:, indKeep, :]
    xc = xc[indKeep, :]
    yc = yc[indKeep, :]
    nMat = nMat[indKeep, :]
    pSLst = [pSLst[k] for k in indKeep]
    # pLLst = [pLLst[k] for k in indKeep]
    pMLst = [pMLst[k] for k in indKeep]
    iInd = iInd[indKeep]
    jInd = jInd[indKeep] 

    # # update from just list of sites to sites per datapoint
    siteIdLst = [siteIdLst[k] for k in jInd] 

    jSite, count = np.unique(jInd, return_counts=True) # sites, # of times site appears
    countAry = np.array([[x, y] for y, x in sorted(zip(count, jSite))]) # rearrange

    # remove points not in target states
    import geopandas as gpd
    from shapely.geometry import Point

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

    THRESHOLD = 30
    nRm = sum(countAry[:, 1] < THRESHOLD)
    indSiteOverThresh = countAry[nRm:, 0].astype(int) 
    indSiteUnderThresh = countAry[:nRm, 0].astype(int) 
    indSiteOverThresh = np.sort(indSiteOverThresh)
    indSiteUnderThresh = np.sort(indSiteUnderThresh)

    # quality
    thresh = 0.4

    # get land cover percentages for quality sites
    lc_idx = [df.varXC.index(var) for var in df.varXC[-6:]]
    lc_pct = df.xc[:, lc_idx]
    lc_pct = lc_pct[indSiteOverThresh]

    top_lc = np.argmax(lc_pct, axis=1)
    land_cover_array_over = np.stack((indSiteOverThresh, top_lc), axis=1)

    mixed_site_idxs = np.where(np.max(lc_pct, axis=1) < thresh)
    land_cover_array_over[mixed_site_idxs, 1] = 5

    from sklearn.model_selection import StratifiedKFold
    data = land_cover_array_over

    dictSubset = dict()

    # Separate the sites and land cover types
    sites = data[:, 0]
    land_cover_types = data[:, 1]

    # Create StratifiedKFold instance
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Split the data
    for k, (train_index, test_index) in enumerate(skf.split(sites, land_cover_types)):
        train_sites, test_sites = sites[train_index], sites[test_index]
        train_land_cover, test_land_cover = land_cover_types[train_index], land_cover_types[test_index]
        
        print(f"Fold {k + 1}")
        print(f"Train sites: {len(train_sites)}")
        # print(f"Train land cover types: {train_land_cover}")
        print(f"Test sites: {len(test_sites)}")
        # print(f"Test land cover types: {test_land_cover}")
        print()

        dictSubset['testSite_k{}5'.format(k)] = test_sites.tolist()
        dictSubset['trainSite_k{}5'.format(k)] = train_sites.tolist()

        indTest = np.where(np.isin(jInd, test_sites))[0]
        indTrain = np.where(np.isin(jInd, train_sites))[0]

        dictSubset['testInd_k{}5'.format(k)] = indTest.tolist()
        dictSubset['trainInd_k{}5'.format(k)] = indTrain.tolist()

    dictSubset['testSite_underThresh'] = indSiteUnderThresh.tolist()
    dictSubset['testInd_underThresh'] = np.where(np.isin(jInd, indSiteUnderThresh))[0].tolist()

    saveFolder = os.path.join(kPath.dirVeg, 'model', 'attention', 'RANDOM_ASSIGNMENT')
    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)
    # dataFile = os.path.join(saveFolder, 'data.npz')
    # np.savez_compressed(dataFile, x=x, xc=xc, y=yc, yc=yc, tInd=iInd, siteInd=jInd)
    subsetFile = os.path.join(saveFolder, 'subset.json')
    with open(subsetFile, 'w') as fp:
        json.dump(dictSubset, fp, indent=4)