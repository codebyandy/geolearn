from mpl_toolkits import basemap
import matplotlib.pyplot as plt
import numpy as np


def mapPoint(ax, lat, lon, data, title=None, vRange=None, cmap=plt.cm.jet, s=30, marker='o'):
    if np.isnan(data).all():
        print('all nan in data')
        return
    if vRange is None:
        vmin = np.percentile(data[~np.isnan(data)], 10)
        vmax = np.percentile(data[~np.isnan(data)], 90)
    else:
        vmin, vmax = vRange
    mm = basemap.Basemap(llcrnrlat=25, urcrnrlat=50,
                         llcrnrlon=-125, urcrnrlon=-65,
                         projection='cyl', resolution='c', ax=ax)
    mm.drawcoastlines()
    mm.drawcountries(linestyle='dashed')
    mm.drawstates(linestyle='dashed', linewidth=0.5)
    cs = mm.scatter(lon, lat, c=data, cmap=cmap,
                    s=s, marker=marker, vmin=vmin, vmax=vmax)
    mm.colorbar(cs, location='bottom', pad='5%')
    ax.set_title(title)
    return mm


def plotTS(ax, t, y, *, styLst=None, tBar=None, cLst='rbkgcmy', legLst=None):
    y = y if type(y) is list else [y]
    for k in range(len(y)):
        yy = y[k]
        sty = '--*' if styLst is None else styLst[k]
        legStr = None if legLst is None else legLst[k]
        ax.plot(t, yy, sty, color=cLst[k], label=legStr)
    if tBar is not None:
        ylim = ax.get_ylim()
        tBar = [tBar] if type(tBar) is not list else tBar
        for tt in tBar:
            ax.plot([tt, tt], ylim, '-k')
    if legLst is not None:
        ax.legend(loc='upper right', frameon=False)
    ax.xaxis_date()
    return ax


def plotCDF(ax, x, cLst='rbkgcmy', legLst=None):
    x = x if type(x) is list else [x]
    for k in range(len(x)):
        xx = x[k]
        xS = sortData(xx)
        yS = np.arange(len(xS)) / float(len(xS) - 1)
        legStr = None if legLst is None else legLst[k]
        ax.plot(xS, yS, color=cLst[k], label=legStr)
    if legLst is not None:
        ax.legend(loc='bottom right', frameon=False)


def plotBox(ax, x, labLst=None, c='r'):
    temp = x
    if type(temp) is list:
        for kk in range(len(temp)):
            tt = temp[kk]
            if tt is not None and tt != []:
                tt = tt[~np.isnan(tt)]
                temp[kk] = tt
            else:
                temp[kk] = []
    else:
        temp = temp[~np.isnan(temp)]
    bp = ax.boxplot(temp, patch_artist=True, notch=True, showfliers=False)
    # ax.set_xticks(range(len(vLst)), vLst)
    if labLst is not None:
        _ = plt.setp(ax, xticks=range(1, len(labLst)+1), xticklabels=labLst)
    for kk in range(0, len(bp['boxes'])):
        _ = plt.setp(bp['boxes'][kk], facecolor=c)
    return ax


def plotHeatMap(ax, mat, labLst, fmt='{:.0f}', vRange = None):
    ny, nx = mat.shape
    ax.set_xticks(np.arange(ny))
    ax.set_yticks(np.arange(nx))
    if ny == nx:
        labX = labLst
        labY = labLst
    else:
        labX = labLst[1]
        labY = labLst[0]
    ax.set_xticklabels(labX)
    ax.set_yticklabels(labY)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    im = ax.imshow(mat)
    for j in range(ny):
        for i in range(nx):
            text = ax.text(i, j, fmt.format(mat[j, i]),
                           ha="center", va="center", color="w")
    return ax


def sortData(x):
    xArrayTemp = x.flatten()
    xArray = xArrayTemp[~np.isnan(xArrayTemp)]
    xSort = np.sort(xArray)
    return xSort
