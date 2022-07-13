import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
from scipy.signal import medfilt
from lmfit.models import LinearModel, LorentzianModel, PolynomialModel

def lamb2shift(lamb,ex):
    shift = 1e7/ex - 1e7/lamb
    return shift

def SNV(I_data):
    mean = I_data.mean()
    std = I_data.std()
    
    return (I_data-mean)/std

def MinMax(I_data):
    mn = I_data.min()
    mx = I_data.max()
    
    return (I_data-mn)/(mx-mn)

def Despike_Norm(I_data):
    despiked = medfilt(I_data, kernel_size=7)
    return MinMax(despiked)

class Sample:
    def __init__(self,hdf_file):
        with h5py.File(hdf_file, 'r') as hdf:
            self.data_file = hdf_file
            self.Details =  {i:v for i,v in hdf.attrs.items()}
            self.Measurements = [i for i in hdf.keys()]
            
    def Laser_reflectivity(self):
        with h5py.File(self.data_file, 'r') as hdf:
            g = hdf['In situ Measurements/Laser Reflectivity']
            data = np.array(g)
            meta = {i:v for i,v in g.attrs.items()}
            
        return data,meta
    
    def Ion_probe(self):
        with h5py.File(self.data_file, 'r') as hdf:
            g = hdf['In situ Measurements/Ion Probe']
            data = np.array(g)
            meta = {i:v for i,v in g.attrs.items()}
        return data,meta
    
    def In_situ_Raman(self):
        with h5py.File(self.data_file, 'r') as hdf:
            g = hdf['In situ Measurements/Raman']
            data = np.array(g)
            meta = {i:v for i,v in g.attrs.items()}
        
        return data,meta
    
    def ICCD_images(self):
        with h5py.File(self.data_file, 'r') as hdf:
            g = hdf['In situ Measurements/ICCD Imaging']
            data = np.array(g)
            meta = {i:v for i,v in g.attrs.items()}
            
            return data,meta
        

        
def get_WSe2_A1_FWHM_Si_ref_Raman(sample_h5_file_abs_path, reference_file_csv_abs_path,visualize=False):
    
    # Get Raman spectrum from sample hdf5 file, crop, despike, and normalize.
    sample = Sample(sample_h5_file_abs_path)
    ram_data, meta = sample.In_situ_Raman()
    w0, I0 = ram_data[:,4], ram_data[:,5]
    w, I = lamb2shift(w0,meta['Excitation Wavelength (nm)']), I0
    bounds = (w>=75)&(w<=700)
    sub_x, sub_y = w[bounds], Despike_Norm(I[bounds])
    
    #This is the cropped, despiked, and normalized Raman data
    xdat, ydat = sub_x, sub_y
    
    # Crop, despike, and normalize the reference Si spectrum
    ref_df = pd.read_csv(reference_file_csv_abs_path)
    w0ref, I0ref = ref_df['Wavelength'], ref_df['Intensity']
    w, I = lamb2shift(w0ref,meta['Excitation Wavelength (nm)']), I0ref
    sub_x, sub_y = w[bounds], I[bounds]
    
    #This is the cropped, despiked and normalized Si substrate spectrum
    subtrate_x, substrate_y = sub_x, Despike_Norm(sub_y)
    
    # Substrate subtracted data
    I_substrate_subtracted = ydat-substrate_y
    
    # Create the Raman spectrum multi Lorentzian model
    model = PolynomialModel(5, prefix='bkg_',nan_policy='omit')
    params = model.make_params(c0=0,c1=0,c2=0,c3=0,c4=0,c5=0)
    
    def add_peak(prefix, center, amplitude=1, sigma=10):
        peak = LorentzianModel(prefix=prefix)
        pars = peak.make_params()
        pars[prefix + 'center'].set(center,min=center-10,max=center+10)
        pars[prefix + 'amplitude'].set(amplitude,min=0)
        pars[prefix + 'sigma'].set(sigma,min=2.5,max=40)            
        return peak, pars

    rough_peak_positions = [117.4,230.1,247.3,364]#[LA2M_guess[i],351]
    for j, cen in enumerate(rough_peak_positions):
        peak, pars = add_peak('peak_%d_' % (j), cen)
        model = model + peak
        params.update(pars)   
    
    # Fit the model
    init = model.eval(params, x=xdat)
    result = model.fit(I_substrate_subtracted, params, x=xdat, max_nfev = 10000)
    comps = result.eval_components()
    
    if result.params['peak_2_fwhm'].stderr == None:
        print('This is probably a bad fit, check and retry')
        
    if visualize:
        plt.plot(xdat, I_substrate_subtracted, label='data')
        plt.plot(xdat, result.best_fit, label='best fit')
        for name, comp in comps.items():
            if name != 'bkg_':
                plt.plot(xdat, comp+comps['bkg_'], '--', label=name)
            else:
                plt.plot(xdat,comp,'--',label=name)
        plt.show()
                
    return result
    