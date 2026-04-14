import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import least_squares
from scipy.special import erf

def measure_gamma(log_exposure, density_curves, density_0=0.25, density_1=1.0):
    gamma = np.zeros((3,))
    for i in range(3):
        loge0 = interp1d(density_curves[:, i], log_exposure, kind='cubic')(density_0)
        loge1 = interp1d(density_curves[:, i], log_exposure, kind='cubic')(density_1)
        gamma[i] = (density_1-density_0)/(loge1-loge0)
    return gamma

def measure_slopes_at_exposure(log_exposure, density_curves, 
                               log_exposure_reference=0.0,
                               log_exposure_range=np.log10(2**2)):
    le_ref = log_exposure_reference
    log_exposure_0 = le_ref - log_exposure_range/2
    log_exposure_1 = le_ref + log_exposure_range/2
    gamma = np.zeros((3,))
    for i in range(3):
        sel = ~np.isnan(density_curves[:,i])
        density_1 = CubicSpline(log_exposure[sel], density_curves[sel,i])(log_exposure_1)
        density_0 = CubicSpline(log_exposure[sel], density_curves[sel,i])(log_exposure_0)
        gamma[i] = (density_1-density_0)/(log_exposure_1-log_exposure_0)
    return gamma

def measure_density_min(log_exposure, density_curves, info_type, control_plot=False):
    
    dc = density_curves
    le = log_exposure
    density_min = np.zeros(3)
    
    # # fitting model for the toe
    # def curve_toe(e, k):
    #     gamma = k[0]
    #     e0    = k[1]
    #     d0    = k[2]
    #     c1    = k[3] 
    #     y = (  gamma/c1 * np.log10(1 + 10**(c1 * (e - e0) ) ) 
    #         ) + d0
    #     return y
    
    def curve_toe(e, k):
        d_max = k[0]
        e0    = k[1]
        d0    = k[2]
        c1    = k[3] 
        if info_type=='positive':
            c1 = -c1
        y = (  d_max * (1+erf(c1*(e-e0)))/2 + d0)
        return y
    # starting parameters of fit
    if info_type=='positive':
        k0 = (  2,  0, 0.05,  2)
        lb = (1.5, -2,  0.0,  0)
        ub = (  4,  2,  0.5,  4)
        fraction_to_fit = 0.04
    else:
        k0 = (  2,  0, 0.05,  2)
        lb = (  1, -2,    0,  0)
        ub = (  4,  2,    1,  4)
        fraction_to_fit = 0.1
    
    fits = []  
    # fit all the toes and save density min
    for i in np.arange(3):
        data = np.copy(dc[:,i])
        # mask data above the toe (1/10 of density range)
        data[data>((np.nanmax(data)-np.nanmin(data))*fraction_to_fit + np.nanmin(data))] = np.nan
        def residues(k):
            res = data - curve_toe(le, k)
            res = np.nan_to_num(res)
            return res
        fit = least_squares(residues, k0, bounds=(lb, ub))
        fits.append(fit)
        k = fit.x
        density_min[i] = k[2]
        
    if control_plot:
        import matplotlib.pyplot as plt
        _, axs = plt.subplots(1,3,figsize=(15,5))
        for i in range(3):
            axs[i].plot(le, dc[:,i], 'o', label='data')
            le_fit = np.linspace(np.nanmin(le), np.nanmax(le), 100)
            dc_fit = curve_toe(le_fit, fits[i].x)
            axs[i].plot(le_fit, dc_fit, '-', label='fit')
            axs[i].axhline(density_min[i], color='r', linestyle='--', label='density min')
            axs[i].set_xlabel('log exposure')
            axs[i].set_ylabel('density')
            axs[i].set_title(f'Channel {i}')
            axs[i].set_ylim([0, 1.5])
            axs[i].set_xlim((-3,3))
            axs[i].legend()
        plt.show()
    return density_min