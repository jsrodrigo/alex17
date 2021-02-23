import pandas as pd
import numpy as np
import xarray as xr
import rasterio as rio
import matplotlib.pyplot as plt
from pandas import IndexSlice as idx
import dataframe_image as dfi
import seaborn as sns
from scipy import stats
# from matplotlib.colors import LightSource
from lib.common_functions import vector_WD_stats, vector_WD_diff

def save_dftable(saveTables, df, outputpath, filename, png=True):
    if saveTables:
        dfstyle = df.style.background_gradient()     # set the stlyle / color of the table
        df.to_csv(outputpath / (filename +'.csv'))
        if png:
            dfstyle.export_png( str(outputpath / (filename +'.png') ) )

def get_errors(masts_obs, masts_sim):
    bias, mae = {}, {}
    for i_sim in masts_sim:
        sim = masts_sim[i_sim].interp(height=masts_obs.height, time=masts_obs.time )
        # bias so far defined as observations - simualations
        bias[i_sim] = masts_obs - sim
        # overwrite wind direction difference to consider it as a circular quantity
        bias[i_sim]['wind_direction'] = vector_WD_diff(sim['wind_direction'] , masts_obs['wind_direction'])
        # mean absolute error
        mae[i_sim] = np.abs(bias[i_sim])
    return bias, mae

#%%  ----------------------- TABLES -------------------------------

def table_statistics_mast(events, masts_obs, height):
    qoi = ['wind_speed', 'wind_direction', 'turbulence_intensity','wind_shear','stability']

    # create columns and index for dataframe
    index = pd.MultiIndex.from_product([events.keys(), qoi],  names=['event', 'variable'])
    columns = pd.MultiIndex.from_product([masts_obs.id.values, ['mean', 'std']],  names=['mast', ''])

    # initialize the DataFrame
    df  = pd.DataFrame(np.nan, index=index, columns=columns)
    ds = masts_obs.sel(height = height, method='nearest')
    for e in events.keys():
        tSlice = slice(events[e][0],events[e][1])
        for q in qoi:
            for m in ds.id.values:

                da = ds[q].sel(id = m, time = tSlice)
                if q == 'wind_direction':
                    df.loc[idx[e,q], idx[m,'mean']], df.loc[idx[e,q], idx[m,'std']] = vector_WD_stats(da)
                else:
                    if q == 'stability':
                        # in case of stability, we hardcoded to get statistics only at the 10m height level
                        da = masts_obs.sel(height = 10, method='nearest')[q].sel(id = m, time = tSlice)
                    df.loc[idx[e,q], idx[m,'mean']] = da.mean(skipna = True)
                    df.loc[idx[e,q], idx[m,'std']]  = da.std( skipna = True)
    return df


def table_availability_mast(ds, outputpath=None):
    t = ds.time
    nT = ( (t[-1]-t[0])/np.diff(t)[0] ).values   # maximum ideal numer of data records per sensors

    dsn = ds.notnull().sum(dim='time')/nT   # ds.count
    df = dsn.to_dataframe(dim_order=['id', 'height'])

    if outputpath is not None:
        save_dftable(True, df, outputpath, 'Data_availability', png=True)
        # sns.heatmap(df.T, annot=True)
    return df


def table_overall_errors(events, masts_obs, masts_sim, qoi, outputpath=None):
    bias, mae = get_errors(masts_obs, masts_sim)

    index = pd.MultiIndex.from_product([events.keys(), qoi],  names=['event', 'variable'])
    # col   = pd.MultiIndex.from_product([masts_sim.keys(), ['bias', 'mae']],  names=['sim', ''])
    col = masts_sim.keys()

    # initialize DataFrame
    df_bias = pd.DataFrame(np.nan, index=index, columns=col); df_mae = pd.DataFrame(df_bias)
    for s in masts_sim:
        for e in events:
            ibias = bias[s].sel(time = slice(events[e][0],events[e][1]))
            imae  =  mae[s].sel(time = slice(events[e][0],events[e][1]))

            for q in qoi:
                if q == 'wind_direction':
                    df_bias.loc[idx[e,q], s], _ = vector_WD_stats(ibias[q])
                    df_mae.loc[idx[e,q], s], _ = vector_WD_stats(imae[q])
                else:
                    df_bias.loc[idx[e,q], s] = ibias[q].mean(skipna=True)
                    df_mae.loc[idx[e,q], s] = imae[q].mean(skipna=True)

    if outputpath is not None:
        # save_dftable(True, df, outputpath, 'Overall_results')
        plt.figure();sns.heatmap(df_bias, annot=True)
        plt.figure();sns.heatmap(df_mae, annot=True)

    return df_bias, df_mae


def table_errors_by_events(events, masts_obs, masts_sim, qoi, df_outputpath=None):
    bias, mae = get_errors(masts_obs, masts_sim)

    index  = pd.MultiIndex.from_product([events.keys(), qoi],  names=['event', 'variable'])
    col_h  = pd.MultiIndex.from_product([masts_obs.id.values, masts_obs.height.values],  names=['mast', 'heights'])
    col_s  = pd.MultiIndex.from_product([masts_obs.id.values, masts_sim.keys()],  names=['mast', 'sim'])

    # initialize DataFrames
    df_bias_s = pd.DataFrame(np.nan, index=index, columns=col_s);  df_mae_s  = pd.DataFrame(df_bias_s)

    d_bias, d_mae = {}, {}
    for s in masts_sim:
        df_bias_h = pd.DataFrame(np.nan, index=index, columns=col_h)
        df_mae_h  = pd.DataFrame(df_bias_h)

        for e in events:
            ibias = bias[s].sel(time = slice(events[e][0],events[e][1]))
            imae  =  mae[s].sel(time = slice(events[e][0],events[e][1]))

            for m in masts_obs.id.values:
                for q in qoi:
                    if q == 'wind_direction':
                        for h in ibias.height:
                            df_bias_h.loc[idx[e,q],idx[m,h]],_ = vector_WD_stats(ibias[q].sel(id=m, height=h))
                            df_mae_h.loc[idx[e,q], idx[m,h]],_ = vector_WD_stats(imae[q].sel(id=m, height=h))
                        df_bias_s.loc[idx[e,q],idx[m,s]] = vector_WD_stats(ibias[q].sel(id=m))
                        df_mae_s.loc[idx[e,q], idx[m,s]] = vector_WD_stats(imae[q].sel(id=m))
                    else:
                        df_bias_h.loc[idx[e,q], idx[m,:]] = ibias[q].sel(id=m).mean(dim='time', skipna=True)
                        df_mae_h.loc [idx[e,q], idx[m,:]] = imae[q].sel(id=m).mean(dim='time', skipna=True)
                        df_bias_s.loc[idx[e,q], idx[m,s]] = ibias[q].sel(id=m).mean(skipna=True)
                        df_mae_s.loc [idx[e,q], idx[m,s]] = imae[q].sel(id=m).mean(skipna=True)

        d_bias[s], d_mae[s] = df_bias_h, df_mae_h

        if df_outputpath is not None:
            save_dftable(True, df_bias_h, df_outputpath, 'Bias_h_'+s)
            save_dftable(True, df_mae_h, df_outputpath,  'Mae_h_'+s)

    if df_outputpath is not None:
        save_dftable(True, df_bias_s, df_outputpath, 'Overall_bias', False)
        save_dftable(True, df_mae_s, df_outputpath,  'Overall_mae', False)

    return df_bias_s, df_mae_s, d_bias, d_mae


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
        statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, ts.values.flatten(),
                                                                        statistic=statistic,
                                                                        bins=bins, expand_binnumbers = True)
        N_WDzL = pd.DataFrame(statistic, index=WDbins_label, columns=zLbins_label)

        binmap = np.empty((Nwd, NzL), dtype = object)
        for i_wd in range(Nwd):
            for i_zL in range(NzL):
                binmap[i_wd,i_zL] = ts[np.logical_and(binnumber[0,:] == i_wd+1, binnumber[1,:] == i_zL+1)].time.values

        if plot:
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
