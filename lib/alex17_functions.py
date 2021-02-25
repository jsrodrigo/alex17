import pandas as pd
import numpy as np
import datetime
import netCDF4
import utm
import xarray as xr
import math
import rasterio as rio
import matplotlib.pyplot as plt 
from matplotlib.colors import LightSource
from scipy import stats
import seaborn as sns

from lib.variables_dictionary.variables import Variables
from lib.variables_dictionary.variables import nc_global_attributes_from_yaml

def annotate_bars(ax,data,col,color = 'k'):
    """
    Add annotations in horizontal bar plots
    Inputs:
        - ax: axis handle
        - data: dataframe with data in bar plot
        - col: column of dataframe to annotate
        - color: text color
    """
    cnt = 0
    for index, row in data.iterrows():
        if row[col]<0: 
            ha = 'left'
        else:
            ha = 'right'
        ax.text(row[col],cnt+0.25, round(row[col],2), color=color, ha=ha)
        cnt = cnt + 1
        
def read_sim(filename):
    try:
        M = xr.open_dataset(filename)
        print(filename)
        try: 
            M = M.resample(time = "1H").mean() # Resample to hourly data
            Umean = M.eastward_wind
            Vmean = M.northward_wind
            Smean = (Umean**2 + Vmean**2)**0.5
            WDmean = 180 + np.arctan2(Umean, Vmean) * 180/np.pi
            M = M.assign(wind_speed = Smean, wind_direction = WDmean)
            if 'simID' in list(M.coords.keys()):
                M = M.drop('simID')
        except AttributeError:
            M = M.assign(eastward_wind = -M.wind_speed*np.sin(M.wind_direction*np.pi/180))
            M = M.assign(northward_wind = -M.wind_speed*np.cos(M.wind_direction*np.pi/180))
            M = M.resample(time = "1H").mean() # Resample to hourly data
            WDmean = 180 + np.arctan2(M.eastward_wind, M.northward_wind) * 180/np.pi 
            M = M.assign(wind_direction = WDmean)
    except IOError:
        M = np.nan
        print(filename + ' is missing')
        
    return M

def read_obs(filename):
    M = xr.open_dataset(filename)
    #M = M.where(M <= 360) # nan when is greater than 360
    M = M.rename({'wind_from_direction': 'wind_direction'})
    M = M.rename({'specific_turbulent_kinetic_energy': 'turbulent_kinetic_energy'})
    M = M.rename({'specific_upward_sensible_heat_flux_in_air': 'heat_flux'})
    M = M.interpolate_na(dim='height') # Fill nans with interpolated values between observational levels
    M = M.assign(eastward_wind = -M.wind_speed*np.sin(M.wind_direction*np.pi/180))
    M = M.assign(northward_wind = -M.wind_speed*np.cos(M.wind_direction*np.pi/180))
    M = M.resample(time = "1H").mean() # Resample to hourly data
    WDmean = 180 + np.arctan2(M.eastward_wind, M.northward_wind) * 180/np.pi # calculate wind direction from U,V components after resampling 
    M = M.assign(wind_direction = WDmean)    
    return M

def vector_mean_std_wind_direction(WD):
    U = -np.sin(2*np.pi*WD/360)
    V = -np.cos(2*np.pi*WD/360)
    Umean = np.mean(U)
    Vmean = np.mean(V)
    WDmean = 180 + math.atan2(Umean,Vmean)*180/np.pi
    eps = np.sqrt(1 - (Umean**2 + Vmean**2))
    WDstd = math.asin(eps)*(1 + (2./np.sqrt(3.) - 1)*eps**3)*180/np.pi # Yamartino (1984)
    return WDmean, WDstd

def events_qois_vs_masts_table(events,masts_obs,height):
    qois = ['wind_speed', 'wind_direction', 'turbulence_intensity',
                            'wind_shear','stability'] 
    events_name = list(events.keys())

    index = pd.MultiIndex.from_product([list(events.keys()), qois],
                                       names=['event', 'QoI'])
    columns = pd.MultiIndex.from_product([masts_obs.id.values.tolist(), ['mean', 'std']],
                                         names=['mast', ''])

    # create the DataFrame
    df = pd.DataFrame(np.random.randn(len(events_name)*len(qois), len(masts_obs.id)*2), index=index, columns=columns)
    idx = pd.IndexSlice
    
    masts = masts_obs.id.values.tolist()

    for e in events_name:
        for q in qois:
            for m in masts:
                if q == 'wind_speed' or q == 'turbulence_intensity':
                    df.loc[idx[e, q], idx[m, 'mean']] = masts_obs[q].sel(id = m, height = height, 
                                                                         time = slice(events[e][0],events[e][1])).mean(skipna = True)
                    df.loc[idx[e, q], idx[m, 'std']] = masts_obs[q].sel(id = m, height = height, 
                                                                        time = slice(events[e][0],events[e][1])).std(skipna = True)
                elif q == 'wind_direction':
                    WD = masts_obs[q].sel(id = m, height = height, time = slice(events[e][0],events[e][1]))
                    df.loc[idx[e, q], idx[m, 'mean']], df.loc[idx[e, q], idx[m, 'std']] = vector_mean_std_wind_direction(WD)
                elif q == 'wind_shear':
                    df.loc[idx[e, q], idx[m, 'mean']] = masts_obs[q].sel(id = m, 
                                                                        time = slice(events[e][0],events[e][1])).mean(skipna = True)
                    df.loc[idx[e, q], idx[m, 'std']] = masts_obs[q].sel(id = m, 
                                                                        time = slice(events[e][0],events[e][1])).std(skipna = True)
                elif q == 'stability':
                    df.loc[idx[e, q], idx[m, 'mean']] = masts_obs[q].sel(id = m, height = 10, 
                                                                         time = slice(events[e][0],events[e][1])).mean(skipna = True)
                    df.loc[idx[e, q], idx[m, 'std']] = masts_obs[q].sel(id = m, height = 10, 
                                                                        time = slice(events[e][0],events[e][1])).std(skipna = True)
    return df 

def WDzL_bins(x,y,ts,statistic,bins,bins_label,plot = False):
        """
        Compute and plot distribution of samples per bin
        Inputs:
            - x: time-series of wind direction
            - y: time-series of stability
            - ts: time-series of values
            - statistic: which statistic to compute per bin
            - bins: bin limits for x and y
            - bins_label: bin labels
            - plot: whether to plot the distribution or not
        Outputs: 
            - N_WDzL: dataframe wind bin sample count per wd and zL
            - binmap: list of timestamp indices to samples in each bin (wd,zL)
        """
        
        Nwd, NzL = [len(dim) for dim in bins_label]
        WDbins_label, zLbins_label = bins_label
        WDbins, zLbins = bins
        x = x.values.flatten()
        x[x>WDbins[-1]] = x[x>WDbins[-1]]-360
        y = y.values.flatten()
        if Nwd == 1:
            statistic, bin_edges, binnumber = stats.binned_statistic(y, ts.values.flatten(), 
                                                                    statistic=statistic, 
                                                                    bins=zLbins)
            N_WDzL = pd.DataFrame(statistic, index=zLbins_label, columns=WDbins_label)
            binmap = np.empty((Nwd, NzL), dtype = object)
            for i_zL in range(NzL):
                binmap[0,i_zL] = ts[binnumber == i_zL+1].time.values
        else:
            statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, ts.values.flatten(), 
                                                                    statistic=statistic, 
                                                                    bins=bins, expand_binnumbers = True)
            N_WDzL = pd.DataFrame(statistic, index=WDbins_label, columns=zLbins_label)    
            binmap = np.empty((Nwd, NzL), dtype = object)
            for i_wd in range(Nwd):
                for i_zL in range(NzL):
                    binmap[i_wd,i_zL] = ts[np.logical_and(binnumber[0,:] == i_wd+1, binnumber[1,:] == i_zL+1)].time.values

        if Nwd > 1 and plot:
            N_zL = np.sum(N_WDzL, axis = 0).rename('pdf')
            N_WD = np.sum(N_WDzL, axis = 1).rename('pdf')
            Nnorm_WDzL = N_WDzL.div(N_WD, axis=0)
            NzL = len(bins_label[1])

            f1 = plt.figure(figsize = (18,8))
            cmap = plt.get_cmap('bwr')
            zLcolors = np.flipud(cmap(np.linspace(0.,NzL,NzL)/NzL))
            ax1=Nnorm_WDzL.plot.bar(stacked=True, color=zLcolors, align='center', width=1.0, legend=False, 
                                    rot=90, use_index = False, edgecolor='grey')
            ax2=(N_WD/N_WD.sum()).plot(ax=ax1, secondary_y=True, style='k',legend=False, rot=90, use_index = False);
            ax2.set_xticklabels(WDbins_label)
            #ax1.set_title('Wind direction vs stability')
            ax1.set_ylabel('$pdf_{norm}$($z/L_{ref}$)')
            ax2.set_ylabel('$pdf$($WD_{ref}$)', rotation=-90, labelpad=15)
            ax1.set_yticks(np.linspace(0,1.,6))
            ax1.set_ylim([0,1.])
            ax2.set_yticks(np.linspace(0,0.2,6))

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            plt.legend(h1+h2, l1+l2, bbox_to_anchor=(1.3, 1))

            #cellText = N_WDzL.T.astype(str).values.tolist() # add table to bar plot
            #the_table = plt.table(cellText=cellText,
            #                      rowLabels=zLbins_label,
            #                      rowColours=zLcolors,
            #                      colLabels=WDbins_label,
            #                      loc='bottom')

        return N_WDzL, binmap

def dataset_binavrg(dataset,binmap,bins_label):
    dims = ['id','height','wd','zL']
    labels = [dataset.id.values] + [dataset.height.values] + bins_label
    variables = list(dataset.keys())

    Nbin = [len(dim) for dim in labels]
    dd = dict.fromkeys(dims, None)
    for d in range(len(dims)):
        dd[dims[d]] = labels[d]

    Nwd, NzL = [len(dim) for dim in bins_label]
    meandata = np.empty([len(variables)] + Nbin)*np.nan
    for i_wd in range(Nwd):
        for i_zL in range(NzL):
            meandata[:,:,:,i_wd,i_zL] = dataset.sel(time = binmap[i_wd,i_zL]).mean(dim = 'time', skipna=True).to_array()

    dv = dict((v,(dims,meandata[iv,:,:,:,:])) for iv,v in enumerate(variables))
    xr_mean = xr.Dataset(data_vars = dv, coords = dd)  
    
    return xr_mean

def xarray_init(dims,labels,variables):
    """
    Initialize xarray
    Inputs: 
        - dims: list with dimmension names
        - labels: coordinates of each dimmension (bins labels)
        - variables: list with variable names
    outputs: 
        - xr_init: empty xarray 
    """
    Nbin = [len(dim) for dim in labels]
    dd = dict.fromkeys(dims, None)
    dv = dict((v,(dims,np.empty(Nbin)*np.nan)) for v in variables)
    for d in range(len(dims)):
        dd[dims[d]] = labels[d]
    xr_init = xr.Dataset(data_vars = dv, coords = dd)     
    return xr_init

def _l(j):
    # a relatively easy way to set line' specs for all the plots [color, style, width, marker]
    lspecs = [['mediumvioletred', '#16A085', 'mediumblue','#D4AC0D','firebrick','#808B96','#5DADE2','C2','#A569BD',
               '#641E16', '#A93226','#D98880','#ABB2B9','C0','C1','C3','C4','C5','C6','C7'],  #
              ['-','--','-.','-',':','-','--',':','-.','-','-','-.','--',':','-','--',':','-.'],
              [1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,2,2,1,1,2,2,1,1.1,1.3,1],
              [None,'*',None,None,None,None,None,None,None,None,None,None,None,None,None,]
              ]
    return lspecs[j]

def mast_sims_vs_obs_timeseries_plot(mast, h, masts_obs, masts_sim, sims, datefrom, dateto, events, binmap):
    WD = masts_obs.wind_direction
    WD = WD.where(WD < 180, WD - 360) # WD relative to North
    fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,1,figsize = (14,14), sharex = True)
    masts_obs.wind_speed.sel(id = mast, height = h).plot(x = 'time', label = 'obs', color = 'k', ax = ax1)
    WD.sel(id = mast, height = h).plot(x = 'time', label = 'obs', color = 'k', ax = ax2)
    masts_obs.turbulence_intensity.sel(id = mast, height = h).plot(x = 'time', label = 'obs', color = 'k', ax = ax3)
    masts_obs.wind_shear.sel(id = mast).plot(x = 'time', label = 'obs', color = 'k', ax = ax4)
    masts_obs.stability.sel(id = mast, height = 10).plot(x = 'time', label = 'obs', color = 'k', ax = ax5)
    masts_obs.heat_flux.sel(id = mast, height = 10).plot(x = 'time', label = 'obs', color = 'k', ax = ax6)
    for i_sim in range (0,len(masts_sim)): 
        WD = masts_sim[i_sim].wind_direction
        WD = WD.where(WD < 180, WD - 360) # WD relative to North
        
        masts_sim[i_sim].wind_speed.sel(id = mast).interp(height= h).plot(x = 'time', label = sims['Label'][i_sim], ax = ax1,
                                                                         color=_l(0)[i_sim], ls=_l(1)[i_sim], lw=_l(2)[i_sim], marker=_l(3)[i_sim])
        WD.sel(id = mast).interp(height= h).plot(x = 'time', label = sims['Label'][i_sim], ax = ax2,
                                                                             color=_l(0)[i_sim], ls=_l(1)[i_sim], lw=_l(2)[i_sim], marker=_l(3)[i_sim])
        masts_sim[i_sim].turbulence_intensity.sel(id = mast).interp(height= h).plot(x = 'time', label = sims['Label'][i_sim], ax = ax3,
                                                                                   color=_l(0)[i_sim], ls=_l(1)[i_sim], lw=_l(2)[i_sim], marker=_l(3)[i_sim])
        masts_sim[i_sim].wind_shear.sel(id = mast).plot(x = 'time', label = sims['Label'][i_sim], ax = ax4,
                                                       color=_l(0)[i_sim], ls=_l(1)[i_sim], lw=_l(2)[i_sim], marker=_l(3)[i_sim])
        
    if binmap.size != 0:
        color_bins = ['salmon', 'silver', 'lightblue']
        n_wd, n_zL = binmap.shape
        for zL in range(n_zL):
            for b in binmap[0,zL]:
                ax1.axvspan(b, b + np.timedelta64(1, 'h'), alpha=0.3, color=color_bins[zL], linewidth = 0) 
                ax2.axvspan(b, b + np.timedelta64(1, 'h'), alpha=0.3, color=color_bins[zL], linewidth = 0) 
                ax3.axvspan(b, b + np.timedelta64(1, 'h'), alpha=0.3, color=color_bins[zL], linewidth = 0) 
                ax4.axvspan(b, b + np.timedelta64(1, 'h'), alpha=0.3, color=color_bins[zL], linewidth = 0) 
                ax5.axvspan(b, b + np.timedelta64(1, 'h'), alpha=0.3, color=color_bins[zL], linewidth = 0) 
                ax6.axvspan(b, b + np.timedelta64(1, 'h'), alpha=0.3, color=color_bins[zL], linewidth = 0) 

    if not events:
        color_events = {'all': 'None', 'neutral': 'silver', 'unstable': 'salmon','stable': 'lightblue', 'very stable': 'deepskyblue'}
        for e in events:
            ax1.axvspan(events[e][0], events[e][1], alpha=0.7, color=color_events[e])
            ax2.axvspan(events[e][0], events[e][1], alpha=0.7, color=color_events[e])
            ax3.axvspan(events[e][0], events[e][1], alpha=0.7, color=color_events[e])
            ax4.axvspan(events[e][0], events[e][1], alpha=0.7, color=color_events[e])
            ax5.axvspan(events[e][0], events[e][1], alpha=0.7, color=color_events[e])
            ax6.axvspan(events[e][0], events[e][1], alpha=0.7, color=color_events[e])
                
    ax1.set_xlim([datefrom, dateto])
    ax1.legend(bbox_to_anchor=(1.13, 1)) 
    ax1.grid(); ax1.set_xlabel(''); ax1.set_ylabel(r'S [$m s^{-1}$]'); ax1.set_title("{} ({} m)".format(mast, h))
    ax2.grid(); ax2.set_xlabel(''); ax2.set_ylabel(r'WD [º]'); ax2.set_title('')
    ax3.grid(); ax3.set_xlabel(''); ax3.set_ylabel(r'TI'); ax3.set_ylim([0,0.7]); ax3.set_title('')
    ax4.grid(); ax4.set_xlabel(''); ax4.set_ylabel(r'$\alpha (80/40)$'); ax4.set_ylim([-0.5,0.5]); ax4.set_title('')
    ax5.grid(); ax5.set_xlabel(''); ax5.set_ylabel(r'$z/L$ ($z$=10m)'); ax5.set_ylim([-2,2]); ax5.set_title('')
    ax6.grid(); ax6.set_xlabel(''); ax6.set_ylabel(r'wT [$m K s^{-1}$]'); ax6.set_title('')# ax6.set_ylim([-1,1])
    return [ax1, ax2, ax3, ax4, ax5, ax6]

def compare_masts_timeseries_plot(mast, h, masts_obs, datefrom, dateto, events):
    fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,1,figsize = (14,14), sharex = True)
    WD = masts_obs.wind_direction
    WD = WD.where(WD < 180, WD - 360) # WD relative to North
    masts_obs.wind_speed.sel(id = mast, height = h).plot(x = 'time', hue = 'id', ax = ax1)
    WD.sel(id = mast, height = h).plot(x = 'time', hue = 'id', ax = ax2)
    masts_obs.turbulence_intensity.sel(id = mast, height = h).plot(x = 'time', hue = 'id', ax = ax3)
    masts_obs.wind_shear.sel(id = mast).plot(x = 'time', hue = 'id', ax = ax4)
    masts_obs.stability.sel(id = mast, height = 10.).plot(x = 'time', hue = 'id', ax = ax5)
    masts_obs.heat_flux.sel(id = mast, height = 10.).plot(x = 'time', hue = 'id', ax = ax6)
    color_events = {'all': 'None', 'neutral': 'silver', 'unstable': 'salmon','stable': 'lightblue', 'very stable': 'deepskyblue'}
    for e in events:
        ax1.axvspan(events[e][0], events[e][1], alpha=0.5, color=color_events[e])
        ax2.axvspan(events[e][0], events[e][1], alpha=0.5, color=color_events[e])
        ax3.axvspan(events[e][0], events[e][1], alpha=0.5, color=color_events[e])
        ax4.axvspan(events[e][0], events[e][1], alpha=0.5, color=color_events[e])
        ax5.axvspan(events[e][0], events[e][1], alpha=0.5, color=color_events[e])
        ax6.axvspan(events[e][0], events[e][1], alpha=0.5, color=color_events[e])
    ax1.set_xlim([datefrom, dateto])
    #ax1.legend(bbox_to_anchor=(1.13, 1))
    ax2.get_legend().remove(); ax3.get_legend().remove(); ax4.get_legend().remove(); ax5.get_legend().remove()
    ax1.grid(); ax1.set_xlabel(''); ax1.set_ylabel(r'wind speed [$m s^{-1}$]')
    ax2.grid(); ax2.set_xlabel(''); ax2.set_ylabel(r'wind direction [º]')
    ax3.grid(); ax3.set_xlabel(''); ax3.set_ylabel(r'turbulence intensity [-]'); ax3.set_ylim([0,0.6])
    ax4.grid(); ax4.set_xlabel(''); ax4.set_ylabel(r'wind shear $\alpha(80/40)$'); ax4.set_ylim([-0.5,0.5])
    ax5.grid(); ax5.set_xlabel(''); ax5.set_ylabel(r'stability $z/L$ ($z$=10m)'); ax5.set_ylim([-1,1])
    ax6.grid(); ax6.set_xlabel(''); ax6.set_ylabel(r'heat flux [$m K s^{-1}$]');# ax6.set_ylim([-1,1])
    return [ax1, ax2, ax3, ax4]

def masts_sims_vs_obs_profiles_plot(event, masts_obs, masts_sim, sims):
    fig, axes = plt.subplots(2,4,figsize = (14,8), sharey = True, sharex = True)
    fig.suptitle('Vertical profiles for event (' + event[0].strftime("%m/%d/%Y, %H:%M:%S") + ' - ' + event[1].strftime("%m/%d/%Y, %H:%M:%S") + ')', fontsize=12)
    masts = masts_obs.coords['id'].values.tolist()
    for i, mast in enumerate(masts):
        index = np.unravel_index(i,(2,4))
        ax = axes[index]
        for i_sim in range (0,len(masts_sim)):
            h_sim = masts_sim[i_sim].wind_speed.sel(time = slice(event[0],event[1]), id = mast).mean(dim = 'time', skipna = True).plot(y = 'height', ax = axes[index], label = sims['Label'][i_sim], color=_l(0)[i_sim], ls=_l(1)[i_sim], lw=_l(2)[i_sim], marker=_l(3)[i_sim])
        h_obs = masts_obs.wind_speed.sel(time = slice(event[0],event[1]), id = mast).mean(dim = 'time', skipna = True).plot(y = 'height', ax = axes[index], label = 'obs',marker = 'o', linestyle='none', markerfacecolor = 'silver', markeredgecolor = 'k')
        ax.set_ylabel(''); ax.set_xlabel(''); 
        ax.set_ylim(1,1000)
        ax.grid()
    plt.yscale('symlog')
    axes[(0,0)].set_ylabel('z [m]'); axes[(1,0)].set_ylabel('z [m]')
    axes[(1,0)].set_xlabel('wind speed [$m s^{-1}$]'); axes[(1,1)].set_xlabel('wind speed [$m s^{-1}$]'); 
    axes[(1,2)].set_xlabel('wind speed [$m s^{-1}$]')
    axes[(1,3)].axis('off')
    axes[(1,2)].legend(bbox_to_anchor=(1.13, 1))
    return axes

def masts_sims_vs_obs_binavrg_profiles_plot(select_bin, masts_obs_mean, masts_sim_mean, sims):
    fig, axes = plt.subplots(2,4,figsize = (14,8), sharey = True, sharex = True)
    masts = masts_obs_mean.coords['id'].values.tolist()
    wd_bin, zL_bin = select_bin
    fig.suptitle('Bin-averaged vertical profiles for bin (' + wd_bin + ',' + zL_bin + ')', fontsize=12)
    for i, mast in enumerate(masts):
        index = np.unravel_index(i,(2,4))
        ax = axes[index]
        for i_sim in range (0,len(masts_sim_mean)):
            h_sim = masts_sim_mean[i_sim].wind_speed.sel(id = mast, wd = wd_bin, zL = zL_bin).plot(y = 'height', ax = axes[index], 
                                                                                                   label = sims['Label'][i_sim],
                                                                                                  color=_l(0)[i_sim], ls=_l(1)[i_sim],
                                                                                                   lw=_l(2)[i_sim], marker=_l(3)[i_sim])
        h_obs = masts_obs_mean.wind_speed.sel(id = mast, wd = wd_bin, zL = zL_bin).plot(y = 'height', ax = axes[index], label = 'obs',
                                                        marker = 'o', linestyle='none', markerfacecolor = 'silver', markeredgecolor = 'k')
        ax.set_ylabel(''); ax.set_xlabel(''); 
        ax.set_ylim(1,1000)
        ax.grid()
    plt.yscale('symlog')
    axes[(0,0)].set_ylabel('z [m]'); axes[(1,0)].set_ylabel('z [m]')
    axes[(1,0)].set_xlabel('wind speed [$m s^{-1}$]'); axes[(1,1)].set_xlabel('wind speed [$m s^{-1}$]'); 
    axes[(1,2)].set_xlabel('wind speed [$m s^{-1}$]')
    axes[(1,3)].axis('off')
    axes[(1,2)].legend(bbox_to_anchor=(1.13, 1))
    return axes

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
    return [ax1, ax2]

def basemap_plot(src, masts, Ztransect, ref, ax, coord = 'utm'):
    # Add overviews to raster to plot faster at lower resolution (https://rasterio.readthedocs.io/en/latest/topics/overviews.html)
    #from rasterio.enums import Resampling
    #factors = [2, 4, 8, 16]
    #dst = rio.open('./inputs/DTM_Alaiz_2m.tif', 'r+')
    #dst.build_overviews(factors, Resampling.average)
    #dst.update_tags(ns='rio_overview', resampling='average')
    #dst.close()
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
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
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    ax.legend(handles = [h_masts,h_A,h_B,h_C])
    plt.colorbar(h_topo, ax = ax, cax = cax)
    return [h_masts,h_A,h_B,h_C]

    
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
    
def get_height_index(f_get_column):
    # height<1000m index
    max_height = 1000
    column = f_get_column(42.695, -1.558)
    all_heights = column.height.values
    last_height_id = np.searchsorted(all_heights, max_height) + 1
    height_index = all_heights[:last_height_id]
    return last_height_id, height_index


def create_nc_file(output_dataset, variables_dimensions, variables_indexes, var_dict, variables_to_write):
    for i in range(len(variables_dimensions)):
        _ = var_dict.nc_create_dimension(output_dataset, variables_dimensions[i], variables_indexes[i])
    variables = {}
    for var_name in variables_to_write:
        variables[var_name[0]] = var_dict.nc_create(output_dataset, var_name[0], variables_dimensions)
    return variables

def time_from_str(time_str):
    ###  str: 2018-09-30 00:00,
    ###       012345678901234567890
    year = int(time_str[0:4])
    month = int(time_str[5:7])
    day = int(time_str[8:10])
    hours = int(time_str[11:13])
    minutes = int(time_str[14:16])
    seconds = 0
    m_sec = 0
    # print(year, month, day, hours, minutes,seconds, m_sec)
    dt = datetime.datetime(year, month, day, hours, minutes, seconds, m_sec)

    seconds_since = (dt - datetime.datetime(1970, 1, 1)).total_seconds()
    return seconds_since


def create_alex17_file_1(file_1_path, f_get_column, variables_to_write):
    # initialise the variables dictionary class
    var_dict = Variables()

    # ---------------------------- create indexes
    # load the met mast locations
    met_mast_locations = pd.read_csv('./inputs/validation_profiles_XYZ.csv')
    utm_zone_number = 30
    utm_zone_letter = 'T'

    # height
    last_height_id, height_index = get_height_index(f_get_column)

    # time
    date_from = time_from_str("2018-09-30 00:00")
    date_to = time_from_str("2018-10-04 00:00")
    #date_to = time_from_str("2018-10-01 00:00")
    time_delta = 10 * 60
    #time_delta = 30 * 60
    time_index = list(range(int(date_from), int(date_to), int(time_delta)))

    print('1. simID_zt.nc:')
    # define the output file structure
    output_dataset = netCDF4.Dataset(file_1_path, 'w', format="NETCDF4")

    variables_indexes = [met_mast_locations['Name'].values, height_index, time_index]
    variables_dimensions = ('id', 'height', 'time')
    variables = create_nc_file(output_dataset, variables_dimensions, variables_indexes, var_dict, variables_to_write)
    nc_global_attributes_from_yaml(output_dataset, './config/Marinet.yaml')

    # fill the output file with results
    for index, row in met_mast_locations.iterrows():
        (lat, lon) = utm.to_latlon(float(row['easting[m]']), float(row['northing[m]']), utm_zone_number,
                                   utm_zone_letter)
        column = f_get_column(lat, lon)  # (time, height, variables)
        column = column.interp(time=time_index)  # interpolate into the desired time resolution
        for v_i in range(len(variables_to_write)):
            # (variables_name v_i)(id index, height :, time:) = (time :, height :, variables v_i).T
            variables[variables_to_write[v_i][0]][index, :, :] = column[:, :last_height_id, v_i].T
    output_dataset.close()


def create_alex17_file_2(file_path, f_get_point, variables_to_write):
    print('2. alex17_00a_box.nc:')

    utm_zone_number = 30
    utm_zone_letter = 'T'

    # initialise the variables dictionary class
    var_dict = Variables()

    # ---------------------------- create indexes
    # x - easting[m]
    easting_index = list(range(612000, 622000, 100))
    # y - northing[m]
    northing_index = list(range(4726000, 4736000, 100))
    # height
    height_index = [40, 125]
    #height_index = [40,]
    # time
    date_from = time_from_str("2018-09-30 00:00")
    date_to = time_from_str("2018-10-04 00:00")
    #date_to = time_from_str("2018-10-01 00:00")
    time_delta = 10 * 60
    #time_delta = 30 * 60
    time_index = list(range(int(date_from), int(date_to), int(time_delta)))

    # define the output file structure
    output_dataset = netCDF4.Dataset(file_path, 'w', format="NETCDF4")
    variables_indexes = [time_index, height_index, easting_index, northing_index]
    variables_dimensions = ('time', 'height', 'easting', 'northing')
    variables = create_nc_file(output_dataset, variables_dimensions, variables_indexes, var_dict, variables_to_write)
    nc_global_attributes_from_yaml(output_dataset, './config/Marinet.yaml')

    # fill the output file with results
    for i_x in range(len(easting_index)):
        x = easting_index[i_x]
        for i_y in range(len(northing_index)):
            y = northing_index[i_y]
            (lat, lon) = utm.to_latlon(x, y, utm_zone_number, utm_zone_letter)
            for i_h in range(len(height_index)):
                height = height_index[i_h]
                point = f_get_point(lat, lon, height)  # (time, variables)
                point = point.interp(time=time_index)  # interpolate into the desired time resolution
                for v_i in range(len(variables_to_write)):
                    # (variables_name v_i)('time' :, 'height' i_h, 'easting' i_x, 'northing' i_y) = (time :, variables v_i)
                    variables[variables_to_write[v_i][0]][:, i_h, i_x, i_y] = point[:, v_i]
    output_dataset.close()


def create_alex17_file_3(file_path, f_get_column, variables_to_write):
    print('3. simID_yzt.nc:')
    utm_zone_number = 30
    utm_zone_letter = 'T'

    # initialise the variables dictionary class
    var_dict = Variables()

    # ---------------------------- create indexes
    # y - northing[m]
    sampling_locations = pd.read_csv('./inputs/validation_ZTransect_XYZ.csv')

    # time
    date_from = time_from_str("2018-09-30 00:00")
    date_to = time_from_str("2018-10-04 00:00")
    #date_to = time_from_str("2018-10-01 00:00")
    time_delta = 10 * 60
    #time_delta = 30 * 60
    time_index = list(range(int(date_from), int(date_to), int(time_delta)))

    # height
    last_height_id, height_index = get_height_index(f_get_column)

    # define the output file structure
    output_dataset = netCDF4.Dataset(file_path, 'w', format="NETCDF4")
    variables_indexes = [sampling_locations['Name'].values, height_index, time_index]
    variables_dimensions = ('id', 'height', 'time')
    variables = create_nc_file(output_dataset, variables_dimensions, variables_indexes, var_dict, variables_to_write)
    nc_global_attributes_from_yaml(output_dataset, './config/Marinet.yaml')

    # fill the output file with results
    for index, row in sampling_locations.iterrows():
        (lat, lon) = utm.to_latlon(float(row['easting[m]']), float(row['northing[m]']), utm_zone_number,
                                   utm_zone_letter)
        column = f_get_column(lat, lon)  # (time, height, variables)
        column = column.interp(time=time_index)  # interpolate into the desired time resolution
        for v_i in range(len(variables_to_write)):
            # (variables_name v_i)(id index, height :, time:) = (time :, height :, variables v_i).T
            variables[variables_to_write[v_i][0]][index, :, :] = column[:, :last_height_id, v_i].T
    output_dataset.close()