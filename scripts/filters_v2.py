import numpy as np
from scipy.optimize import curve_fit
from numpy.polynomial.polynomial import Polynomial
from sklearn.decomposition import PCA
from toast.ops import CommonModeFilter, PolyFilter2D


def az_el_model(times, g, a, c, el, az):
    return (g / np.sin(el)) + (a * az) + c

def az_el_correction(times, tod,  el, az):
    # Initial guess for the parameters
    initial_params = [200, 1, 1]
    
    # Fit the Az-El model
    params, params_covariance = curve_fit(lambda times, g, a, c: 
                                          az_el_model(times, g, a, c,  el, az), 
                                          times, tod, p0=initial_params)
    fitted_g, fitted_a, fitted_c = params
    
    # print(f"Fitted parameters: g={fitted_g}, a={fitted_a}, c={fitted_c}")
    
    fitted_model = az_el_model(times, fitted_g, fitted_a, fitted_c,  el, az)
    azel_corrected_data = tod - fitted_model
    return(azel_corrected_data)

def poly_filter(times, tod, order):
    p = Polynomial.fit(times, tod, order)
    # Evaluate the polynomial fit
    poly_fit = p(times)
    poly_corrected_data = tod - poly_fit
    return(poly_corrected_data)

def pca_component_removal(data, n_components=5):
    #data should be 2D array from 1 Obs
    # Standardize the data
    data_standardized = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)

    # Perform PCA
    pca = PCA()
    # Transpose to fit PCA on samples, then transpose back
    data_pca = pca.fit_transform(data_standardized.T).T  

    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    #print("Explained Variance Ratio:", explained_variance)

    # Choose the number of components to remove (e.g., 2)
    n_components_to_remove = n_components

    # Zero out the leading components in the PCA-transformed data
    components_to_remove = np.zeros_like(data_pca)
    components_to_remove[:n_components_to_remove, :] = data_pca[:n_components_to_remove, :]

    # Reconstruct the unwanted components in the original data space
    unwanted_contribution = pca.inverse_transform(components_to_remove.T).T

    # Subtract the unwanted components from the original standardized data
    data_cleaned_standardized = data_standardized - unwanted_contribution

    # De-standardize the cleaned data
    data_cleaned = (data_cleaned_standardized * np.std(data, axis=1, keepdims=True)) + \
                        np.mean(data, axis=1, keepdims=True)

    data_components_removed = (unwanted_contribution * np.std(data, axis=1, keepdims=True)) + \
                            np.mean(data, axis=1, keepdims=True)
    
    return(data_cleaned, data_components_removed)

def filter_chain(data):
    #print(f"Common mode filter...")
    commonmode_filter = CommonModeFilter()
    commonmode_filter.enabled = True  # Toggle to False to disable
    commonmode_filter.apply(data)
    
    for obs_num in range(len(data.obs)):
        obs = data.obs[obs_num]
        az_obs = obs.shared["azimuth"]
        el_obs = obs.shared["elevation"]
        timestamps = obs.shared["times"]
        #print("="*20,"\n", f"Obs Num {obs_num}")
        #print(f"Az-El Correction...")
        #print(f"Poly Correction...")
        for det in obs.all_detectors:
            # print(det)
            tod_det = obs.detdata["signal"][det]
    
            corrected_tod = az_el_correction(timestamps, tod_det, 
                                                  el_obs, az_obs)
            
            obs.detdata["signal"][det] = corrected_tod
            del corrected_tod, tod_det
            
            ####"Poly Correction..."
            #tod_det = obs.detdata["signal"][det]

            #corrected_tod = poly_filter(timestamps, tod_det, 3)

            #obs.detdata["signal"][det] = corrected_tod
            #del corrected_tod, tod_det
    
    #print("="*20,"\n", f"2D Filter ...")
    polyfilter2D = PolyFilter2D()
    polyfilter2D.order = 3 #5
    polyfilter2D.enabled = True
    polyfilter2D.apply(data)

    for obs_num in range(len(data.obs)):
        obs = data.obs[obs_num]
        #print("="*20,"\n", f"Obs Num {obs_num}")
        #print(f"PCA component removal ...")
        tod_alldets = obs.detdata["signal"]
        n_components = 4 #5
        data_cleaned, data_components_removed = pca_component_removal(tod_alldets, n_components)

        for det_idx,det in enumerate(obs.all_detectors):
            obs.detdata["signal"][det] = data_cleaned[det_idx, :]
        del data_cleaned, data_components_removed

