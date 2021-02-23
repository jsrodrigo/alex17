import pandas as pd
import numpy as np
import xarray as xr
import rasterio as rio
from pathlib import Path
import matplotlib.pyplot as plt
from pandas import IndexSlice as idx
from lib.common_functions import vector_WD_stats, vector_WD_diff

def saveFigureFunction(saveFigures, fig_handle, figureOutFile, dpiFig=100):
    if saveFigures:
        fig_handle.savefig(figureOutFile, bbox_inches='tight', format='png', dpi=dpiFig)

def shade_events_and_labels(ax, events, vars2plot_ts, datefrom, dateto):
    fz = 8 # fontsize of vertical-axis labels
    for ii, ia in enumerate(ax):
        [ia.axvspan(events[e][0], events[e][1], color=events[e][2], alpha=0.5) for e in events]
        ia.grid(); ia.set_xlabel(''); ia.set_ylabel(vars2plot_ts[ii][1], fontsize=fz)
    [ax[x].set_title('') for x in range(1,len(vars2plot_ts))]
    ax[0].set_xlim([datefrom, dateto])

def _l(j):
    # a relatively easy way to set line' specs for all the plots [color, style, width, marker]
    lspecs = [['mediumvioletred', '#16A085', 'mediumblue','#D4AC0D','firebrick','#808B96','#5DADE2','C2','#A569BD',
               '#641E16', '#A93226','#D98880','#ABB2B9','C0','C1','C3','C4','C5','C6','C7'],  #
              ['-','--','-.','-',':','-','--',':','-.','-','-','-.','--',':','-','--',':','-.'],
              [1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,2,2,1,1,2,2,1,1.1,1.3,1],
              [None,'*',None,None,None,None,None,None,None,None,None,None,None,None,None,]
              ]
    return lspecs[j]

def _ds2da(ds, var, mast, h):
    if var == 'wind_shear':
        return ds[var].sel(id=mast)
    elif var=='stability':
        return ds[var].sel(id=mast).interp(height=10)
    else:
        return ds[var].sel(id=mast).interp(height=h)


#%% --------------- PLOTS --------------------------------------

def mast_sims_vs_obs_timeseries_plot(mast, h, vars2plot_ts, masts_obs, masts_sim, datefrom, dateto, events):
    fig, a = plt.subplots(len(vars2plot_ts),1,figsize = (14,12), sharex = True)

    for i, i_v in enumerate(vars2plot_ts):
        _ds2da(masts_obs, i_v[0], mast, h).plot(x='time', ax=a[i], label='obs', color='k')
                  # plot(x='time', ax=a[i], label='obs', marker='o', markersize=1, ls='none', color='grey')

        for j, i_sim in enumerate(masts_sim):
            _ds2da(masts_sim[i_sim], i_v[0], mast, h).\
                plot(x='time', ax=a[i], label=i_sim, color=_l(0)[j], ls=_l(1)[j], lw=_l(2)[j], marker=_l(3)[j])

    shade_events_and_labels(a, events, vars2plot_ts, datefrom, dateto)
    a[-2].legend(bbox_to_anchor=(1, 1))  #a[4].set_ylim([-3,3]);  a[5].set_ylim([-1,1])
    return fig, a


def compare_masts_timeseries_plot(mast, h, vars2plot_ts, masts_obs, datefrom, dateto, events):
    fig, a = plt.subplots(len(vars2plot_ts),1,figsize = (14,14), sharex = True)

    for i, i_v in enumerate(vars2plot_ts):
        for j, i_mast in enumerate(mast):
            _ds2da(masts_obs, i_v[0], i_mast, h).plot(x='time', label=i_mast, color=_l(0)[j], ax=a[i])

    shade_events_and_labels(a, events, vars2plot_ts, datefrom, dateto)
    a[-2].legend(); #a[3].set_ylim([-0.5,0.5]); a[4].set_ylim([-1,1]); # a[5].set_ylim([-1,1])
    return fig, a


def masts_sims_vs_obs_profiles_plot(event, var, masts_obs, masts_sim, masts2plot):
    # select the period to plot
    t = slice(event[0],event[1])
    # configure subplots and figure layout
    n_columns_plot = int(np.ceil(len(masts2plot)/2))
    fig, a = plt.subplots(2,n_columns_plot,figsize = (12,8), sharey = True, sharex = True)
    # loop over mast names and simulations
    for i, mast in enumerate(masts2plot):
        ind = np.unravel_index(i,(2,n_columns_plot))
        for i, i_sim in enumerate(masts_sim):
            masts_sim[i_sim][var[0]].sel(time=t, id=mast).mean(dim='time', skipna=True).\
                  plot.line(y='height', ax=a[ind], label=i_sim, color=_l(0)[i], ls=_l(1)[i], lw=_l(2)[i], marker=_l(3)[i])

        masts_obs[var[0]].sel(time=t, id=mast).mean(dim='time', skipna=True).\
                     plot(y='height', ax=a[ind], label='obs', marker='o', ls='none', color='silver')

        a[ind].set_ylabel(''); a[ind].set_xlabel(''); a[ind].set_ylim(1,2000); a[ind].grid()
    plt.yscale('symlog')
    a[0,0].set_ylabel('z [m]'); a[1,0].set_ylabel('z [m]')
    a[1,0].set_xlabel(var[1]); a[1,1].set_xlabel(var[1]); a[1,2].set_xlabel(var[1])
    a[1,n_columns_plot-1].axis('off')
    a[1,n_columns_plot-2].legend(bbox_to_anchor=(1.13, 1))
    return fig, a


def plot_topography(tif_topofile, Ztransect, masts):

    # read nc file in UTM coordinates
    topo = rio.open(tif_topofile)
    # box = [612000 , 622000, 4726000, 4736000]    # evaluation area

    # Define local coordinate system
    ref = 'M5' # reference site to define origin of coordinate system
    ref = masts[masts['Name'] == ref][['easting[m]','northing[m]','elevation[m]']].values[0].tolist()
    masts['x'] = masts['easting[m]'] - ref[0]
    masts['y'] = masts['northing[m]'] - ref[1]
    masts['z'] = masts['elevation[m]'] - ref[2]
    Ztransect['x'] = Ztransect['easting[m]'] - ref[0]
    Ztransect['y'] = Ztransect['northing[m]'] - ref[1]
    Ztransect['z'] = Ztransect['elevation[m]'] - ref[2]
    box_xy = [612000 - ref[0], 622000- ref[0], 4726000 - ref[1], 4736000 - ref[1]]    # evaluation area

    # Plot elevation map and validation sites
    f, a = plt.subplots(1,2,figsize = (15,6))
    basemap = basemap_plot(topo, masts, Ztransect, ref, a[0], coord = 'xy')
    a[0].set_xlim(box_xy[0:2])
    a[0].set_ylim(box_xy[2:4])

    # Plot north-south transect
    Zprofile = Ztransect_plot(masts, Ztransect, a[1])

    return f, a

def Ztransect_sims_vs_obs_plot(t, Ztransect_obs, Ztransect_sim, masts, Ztransect):
    fig, (ax1,ax2) = plt.subplots(2,1, figsize = (8,6), sharex = True)
    Zprofile = Ztransect_plot(masts, Ztransect, ax2)
    ax2.set_title('')
    #for i_sim in range (0,n_sim):
        #Ztransect_sim[i_sim].wind_speed.sel(height = h).plot(x = 'id', label = sims['ID'][i_sim], ax = ax1)
    #ax1.legend(bbox_to_anchor=(1, 1))
    masts_inZ = [] # index of Z_transect position nearest to each mast
    for i, row in masts.iterrows():
        d = np.sqrt((Ztransect['x'] - masts['x'][i])**2 + (Ztransect['y'] - masts['y'][i])**2)
        masts_inZ.append(d[d == d.min()].index[0])
    for x in masts_inZ:
        ax1.axvline(x, color = 'silver', linestyle = '--', zorder = 0)
    return fig, [ax1, ax2]

def basemap_plot(src, masts, Ztransect, ref, ax, coord = 'utm'):
    # Add overviews to raster to plot faster at lower resolution (https://rasterio.readthedocs.io/en/latest/topics/overviews.html)
    #from rasterio.enums import Resampling
    #factors = [2, 4, 8, 16]
    #dst = rio.open('./inputs/DTM_Alaiz_2m.tif', 'r+')
    #dst.build_overviews(factors, Resampling.average)
    #dst.update_tags(ns='rio_overview', resampling='average')
    #dst.close()
    A_ind = Ztransect['Name'].str.contains('A') # Tajonar ridge scan
    B_ind = Ztransect['Name'].str.contains('B') # Elortz valley scan
    C_ind = Ztransect['Name'].str.contains('C') # Alaiz ridge scan
    oview = src.overviews(1)[2] # choose overview (0 is largest, -1 is the smallest)
    topo = src.read(1, out_shape=(1, int(src.height // oview), int(src.width // oview)))
    if coord == 'xy':
        spatial_extent = [src.bounds.left - ref[0], src.bounds.right - ref[0], src.bounds.bottom - ref[1], src.bounds.top - ref[1]]
    else:
        spatial_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    topo_ma = np.ma.masked_where(topo == 0 , topo, copy=True)
#    ls = LightSource(azdeg=315, altdeg=60)
#    rgb = ls.shade(topo_ma, cmap=plt.cm.terrain, blend_mode='overlay')
#    h_topo = ax.imshow(rgb, extent=spatial_extent, vmin=400, vmax=1200)
    h_topo = ax.imshow(topo_ma, cmap = plt.cm.terrain, extent=spatial_extent, vmin=300, vmax=1200)
    if coord == 'xy':
        h_masts = ax.scatter(masts['x'], masts['y'], s = 10, marker='s', c='k', label = 'Masts')
        h_A = ax.scatter(Ztransect[A_ind]['x'], Ztransect[A_ind]['y'], s = 2, marker='.', c='blue', label = 'A-transect')
        h_B = ax.scatter(Ztransect[B_ind]['x'], Ztransect[B_ind]['y'], s = 2, marker='.', c='black', label = 'B-transect')
        h_C = ax.scatter(Ztransect[C_ind]['x'], Ztransect[C_ind]['y'], s = 2, marker='.', c='red', label = 'C-transect')
        for i, txt in enumerate(masts['Name']):
            ax.annotate(txt, (masts['x'][i]+50, masts['y'][i]+50))
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
    else:
        h_masts = ax.scatter(masts['easting[m]'], masts['northing[m]'], s = 10, marker='s', c='k', label = 'Masts')
        h_A = ax.scatter(Ztransect[A_ind]['easting[m]'], Ztransect[A_ind]['northing[m]'], s = 2, marker='.', c='blue', label = 'A-transect')
        h_B = ax.scatter(Ztransect[B_ind]['easting[m]'], Ztransect[B_ind]['northing[m]'], s = 2, marker='.', c='black', label = 'B-transect')
        h_C = ax.scatter(Ztransect[C_ind]['easting[m]'], Ztransect[C_ind]['northing[m]'], s = 2, marker='.', c='red', label = 'C-transect')
        for i, txt in enumerate(masts['Name']):
            ax.annotate(txt, (masts['easting[m]'][i]+50, masts['northing[m]'][i]+50))
        ax.set_xlabel('Easting [m]')
        ax.set_ylabel('Northing [m]')

    ax.set_title('ALEX17 sites')

    ax.legend(handles = [h_masts,h_A,h_B,h_C])
    plt.colorbar(h_topo, ax = ax)
    return [h_masts, h_A, h_B, h_C]


def Ztransect_plot(masts, Ztransect, ax):
    A_ind = Ztransect['Name'].str.contains('A') # Tajonar ridge scan
    B_ind = Ztransect['Name'].str.contains('B') # Elortz valley scan
    C_ind = Ztransect['Name'].str.contains('C') # Alaiz ridge scan
    h_topoZ = (Ztransect['z']-125.).plot.area(stacked=False, color = 'lightgrey', alpha = 1, ax = ax)
    h_topoA = Ztransect[A_ind]['z'].plot.line(style = '.', color = 'blue', ms = 3, ax = ax)
    h_topoB = Ztransect[B_ind]['z'].plot.line(style = '.', color = 'black', ms = 3, ax = ax)
    h_topoC = Ztransect[C_ind]['z'].plot.line(style = '.', color = 'red', ms = 3, ax = ax)
    masts_inZ = [] # index of Ztransect position nearest to each mast
    for i, row in masts.iterrows():
        d = np.sqrt((Ztransect['x'] - masts['x'][i])**2 + (Ztransect['y'] - masts['y'][i])**2)
        masts_inZ.append(d[d == d.min()].index[0])
    for x in masts_inZ:
        ax.axvline(x, color = 'silver', linestyle = '--', zorder = 0)
    ax.set_title('Z-transect profile at 125 m above ground level')
    ax.set_xlabel('Z-transect position')
    ax.set_ylabel('z [m]')
    ax.set_axisbelow(True)
    ax.yaxis.grid(zorder = 0)
    ax.set_ylim([0,1000])
    ax.set_xticks(masts_inZ)
    ax.set_xticklabels(masts['Name'])

    return [h_topoZ, h_topoA, h_topoB, h_topoC]
