import os
import numpy as np
import math

def ensure_dir(directory):
    os.makedirs(directory) if not os.path.exists(directory) else ''

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

