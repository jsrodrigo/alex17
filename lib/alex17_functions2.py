import os
import numpy as np
import math
import xarray as xr

def ensure_dir(directory):
    os.makedirs(directory) if not os.path.exists(directory) else ''

def read_sim(filename):
    ds = xr.open_dataset(filename)
    U = ds.eastward_wind
    V = ds.northward_wind
    ds['wind_speed'] = (U**2 + V**2)**0.5
    ds['wind_direction'] = (270-np.rad2deg(np.arctan2(V,U)))%360

    ds = _add_vars(ds, interp_method = "linear")

    if 'turbulent_kinetic_energy' in ds:
        ds['turbulence_intensity'] = (((2./3.)*ds['turbulent_kinetic_energy'])**0.5)/ds.wind_speed
    else:
        ds['turbulence_intensity'] = ds.wind_speed * np.nan
    return ds

def read_obs(filename):
    ds = xr.open_dataset(filename)
    # Fill nans with interpolated values between observational levels
    # and then for practical purposes, MP5 mast @ 78m and 102m are dropped
    ds = ds.interpolate_na(dim='height')
    ds = ds.drop_sel(height=[78., 102.])

    # Flag wind speed values == 0 as nan
    ds['wind_speed'] = ds['wind_speed'].where(ds['wind_speed'] != 0, np.nan)

    ds = ds.rename({'wind_from_direction': 'wind_direction'})

    ds = _add_vars(ds, 'nearest')
    ds['turbulence_intensity'] = ds.wind_speed_std / ds.wind_speed
    return ds

def _add_vars(ds, interp_method='nearest'):

    if 'specific_turbulent_kinetic_energy' in ds:
        ds = ds.rename({'specific_turbulent_kinetic_energy': 'turbulent_kinetic_energy'})

    if 'specific_upward_sensible_heat_flux_in_air' in ds:
        ds = ds.rename({'specific_upward_sensible_heat_flux_in_air': 'heat_flux'})

    # wind shear from 80 and 40m wind. Note that this leaves the dataarray without the "height" dimension
    ds['wind_shear'] = np.log(ds.wind_speed.interp(height=80., method=interp_method)/ \
                   ds.wind_speed.interp(height=40., method=interp_method))/np.log(80./40.)

    #I decided not to include the  "height" dimension to the wind_shear variable
    # ds['wind_shear']= shear.expand_dims({'height':[80.]}, axis=1)#

    #  stability based on zL parameter
    if 'obukhov_length' in ds:
        ds['stability'] = ds.height.values[np.newaxis,:,np.newaxis] / ds.obukhov_length
    else:
        ds['stability'] = ds.wind_speed * np.nan
    return ds

def vector_WD_stats(WD):
    U = -np.sin(2*np.pi*WD/360)
    V = -np.cos(2*np.pi*WD/360)
    Umean = np.nanmean(U)
    Vmean = np.nanmean(V)
    WDmean = 180 + math.atan2(Umean,Vmean)*180/np.pi
    eps = np.sqrt(1 - (Umean**2 + Vmean**2))
    WDstd = math.asin(eps)*(1 + (2./np.sqrt(3.) - 1)*eps**3)*180/np.pi # Yamartino (1984)
    return WDmean, WDstd

def vector_WD_diff(WD1, WD2):
    U_dif = -np.sin(2*np.pi*WD2/360) - (-np.sin(2*np.pi*WD1/360))
    V_dif = -np.cos(2*np.pi*WD2/360) - (-np.cos(2*np.pi*WD1/360))
    # return 180 + math.atan2(U_dif,V_dif)*180/np.pi
    return 180 + np.arctan2(U_dif,V_dif)*180/np.pi
