import numpy as np
from scipy import interpolate

from astropy import units as un
from astropy.cosmology import Planck15
from py21cmsense import GaussianBeam, Observatory, Observation, PowerSpectrum, hera

n2=28 #the maximum number of k bins which are valid (not None)
def noise_generator(k_21,delta_21,z):
    
    frequency=1420/(1+z)

    sensitivity = PowerSpectrum(
    observation = Observation(
        observatory = Observatory(
            antpos = hera(hex_num=11, separation=14, dl=12.12),
            beam = GaussianBeam(frequency=frequency, dish_size=14),
            latitude=38*np.pi/180.0
            ),
        spectral_index=2.55,
        tsky_ref_freq=300,
        tsky_amplitude=60000
        ),
    k_21=k_21,
    delta_21=delta_21
    )
    
    power_std = np.array(sensitivity.calculate_sensitivity_1d())
    k_noise=np.array(sensitivity.k1d*Planck15.h)

    return k_noise, power_std
        
        
def get_power(filename, nc,n_noise,z0):
    data = filename
    delta1 = filename.brightness_temp[:, :, :]-np.mean(filename.brightness_temp[:, :, :])

    N_tot = nc**3
    V = (100)**3 #the length should be a free parameter
    delta = delta1*V/N_tot

    space_ps = np.abs(np.fft.fftn(delta))
    space_ps *= space_ps

    delta_k = 2*np.pi/100

    k_factor = 1.35
    k_first_bin_ceil = delta_k
    k_max = delta_k*(nc)

    num_bins = 0
    k_floor = 0
    k_ceil = k_first_bin_ceil
    while (k_ceil < k_max):
        num_bins +=1
        k_floor=k_ceil
        k_ceil *=k_factor

    k_ave = np.zeros(num_bins)
    in_bin_ct = np.zeros(num_bins)
    p_box = np.zeros(num_bins)

    for i in range(0,nc):
        k = i
        if i>=nc/2:
            i-=nc
        k_x = i * delta_k
        for j in range(0,nc):
            if (i!=0 and j!=0):
                l = j
                if j>=nc/2:
                    j-=nc
                k_y = j * delta_k
                for kk in range(0,nc//2):
                    m = kk
                    k_z = kk * delta_k
                    k_mag = np.sqrt(k_x**2+k_y**2+k_z**2)
                    ct = 0
                    k_floor = 0
                    k_ceil = k_first_bin_ceil
                    while (k_ceil<=k_max):
                        if k_mag >= k_floor and k_mag<k_ceil:
                            in_bin_ct[ct] += 1
                            p_box[ct] += k_mag**3*space_ps[k][l][m]/(2*np.pi*np.pi*V)
                            k_ave[ct] += k_mag
                            break
                        ct += 1
                        k_floor = k_ceil
                        k_ceil *=k_factor

    kp = []
    pp = []
    err = []
    for ct in range(1,num_bins):
        if in_bin_ct[ct]>0:
            k_ave[ct] /= (in_bin_ct[ct])
            kp.append(k_ave[ct])
            p_box[ct] /= (in_bin_ct[ct])
            pp.append(p_box[ct])
            err.append(p_box[ct]/np.sqrt(in_bin_ct[ct]))
    k = np.array(kp, dtype=float)
    ## unit 1 / Mpc 
    
    foreground_cut = 0.15
    shot_noise_cut = 1.0
    NSplinePoints = 8
    kSplineMin = foreground_cut
    kSplineMax = shot_noise_cut
    kSpline = np.zeros(NSplinePoints)

    for j in range(NSplinePoints):
        kSpline[j] = kSplineMin + (kSplineMax - kSplineMin)*float(j)/(NSplinePoints - 1)
    
    P_21 = np.array(pp, dtype=float)
    
    ###################generate the noise from 21cmSense
    error2=noise_generator(k,P_21,z0)
    
    splined_error=interpolate.splrep(error2[0],np.log10(error2[1]),s=0)
    splined_model = interpolate.splrep(k,np.log10(P_21),s=0)
    
    ErrorPS_val=10**(interpolate.splev(kSpline[:],splined_error,der=0))
    ModelPS_val = 10**(interpolate.splev(kSpline[:],splined_model,der=0))
    
    ModelPS_val=np.where(np.isnan(ModelPS_val),0,ModelPS_val)

    noised_tot=np.zeros((n_noise,NSplinePoints))
    for k in range(n_noise):
        error_g = np.random.normal(0, ErrorPS_val)
        noised_tot[k]=error_g+ModelPS_val
        #noised_tot[k]=np.where(noised_tot[k]<0,0,noised_tot[k])
        
    return noised_tot
