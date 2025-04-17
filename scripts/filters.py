"""
This script contains functions for data processing including slope removal, azimuth-elevation correction, and PCA component removal.
Functions
---------
remove_slope_1d(data, w=10000)
az_el_model(times, g, a, c, el, az)
    Model function for azimuth-elevation correction.
az_el_correction(times, tod, el, az)
    Applies azimuth-elevation correction to time-ordered data.
pca_component_removal(data, pca_axis=0, standardization=False, n_components=5)

Last updated: 26.11.2024
"""

## Imports
import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA

def remove_slope_1d(data, w=10000):
    """
    Removes a linear slope and mean from a 1D NumPy array.
    
    Parameters
    ----------
    data : np.ndarray
        The 1D array to detrend.
    w : int, optional
        Number of points from the start and end to calculate the slope.
        Defaults to 1.

    Returns
    -------
    detrended_data : np.ndarray
        The detrended 1D array.
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D NumPy array.")

    w = max(1, w)  # Ensure at least one point is used
    n = len(data)
    if n < 2 * w:
        raise ValueError("Array is too short for the given window size.")

    # Calculate the slope using averages of the first and last w points
    slope = (np.mean(data[-w:]) - np.mean(data[:w])) / (n - 1)

    # Generate the slope line to subtract
    trend = slope * np.arange(n) + np.mean(data[:w])

    # Detrend the data
    return data - trend

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

def pca_component_removal(data, pca_axis=1, standardization=False, n_components=None):
    """
    Removes common features from a 2D dataset using PCA.
    Updated: 30.11.2024

    Parameters
    ----------
    data : np.ndarray
        2D numpy array where PCA will be applied.
    pca_axis : int
        The axis along which to identify and remove common components.
        - pca_axis=0: Process across rows (e.g., detectors).
        - pca_axis=1: Process across columns (e.g., time samples).
    standardization : bool
        If True, standardizes the data (mean=0, std=1) before applying PCA.
    n_components : int
        Number of principal components to remove.

    Returns
    -------
    data_cleaned : np.ndarray
        The data with the specified components removed.
    data_components_removed : np.ndarray
        The components removed from the data.
    """
    
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")

    if pca_axis not in [0, 1]:
        raise ValueError("pca_axis must be 0 or 1.")

    # Transpose the data if pca_axis=1 (process across columns)
    transpose = False
    if pca_axis == 1:
        data = data.T
        transpose = True

    # Standardize the data if required (always along axis=0)
    if standardization:
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        # Avoid division by zero
        std[std == 0] = 1e-6
        data_standardized = (data - mean) / std
    else:
        data_standardized = data
        mean = 0
        std = 1  # No scaling

    # Apply PCA
    pca = PCA()
    data_pca = pca.fit_transform(data_standardized)

    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    if n_components is None:
        # Remove components explaining more than 0.1% of the variance
        percent_var = [float(x) * 100 for x in explained_variance]
        n_components = sum(v >= 0.1 for v in percent_var)

    # Zero out the leading components
    components_to_remove = np.zeros_like(data_pca)
    components_to_remove[:, :n_components] = data_pca[:, :n_components]

    # Reconstruct the unwanted components in the original space
    unwanted_contribution = pca.inverse_transform(components_to_remove)

    # Subtract the unwanted components
    data_cleaned_standardized = data_standardized - unwanted_contribution

    # De-standardize the cleaned data
    data_cleaned = (data_cleaned_standardized * std) + mean
    # data_components_removed = (unwanted_contribution * std) + mean

    # Transpose back if necessary
    if transpose:
        data_cleaned = data_cleaned.T
        # data_components_removed = data_components_removed.T

    # return (data_cleaned, data_components_removed, explained_variance)
    return (data_cleaned, explained_variance, n_components)

def remove_global_poly(data_2d, times, degree=3, bin_size=1000):
    """
    Removes a global polynomial trend from each row of a 2D NumPy array by downsampling.

    Parameters
    ----------
    data_2d : np.ndarray
        The 2D array where each row is detrended individually.
    times : np.ndarray
        The time or index array of the same length as the columns of `data_2d`.
    degree : int, optional
        The degree of the polynomial to fit and subtract. Defaults to 10.
    bin_size : int, optional
        The number of data points to average for downsampling. Defaults to 1000.

    Returns
    -------
    corrected_data : np.ndarray
        The 2D array with polynomial trends removed from each row.
    """
    if data_2d.ndim != 2:
        raise ValueError("Input data must be a 2D NumPy array.")
    if len(times) != data_2d.shape[1]:
        raise ValueError("Length of `t0` must match the number of columns in `data_2d`.")
    
    corrected_data = np.empty_like(data_2d)

    for i, data in enumerate(data_2d):
        # Downsample t0 and data
        num_bins = len(times) // bin_size
        t0_downsampled = times[:num_bins * bin_size].reshape(-1, bin_size).mean(axis=1)
        data_downsampled = data[:num_bins * bin_size].reshape(-1, bin_size).mean(axis=1)
        
        # Fit a polynomial to the downsampled data
        p = np.polynomial.Polynomial.fit(t0_downsampled, data_downsampled, degree)
        
        # Evaluate the trend across the full original data
        trend = p(times)

        # Subtract the polynomial trend
        corrected_data[i] = data - trend
    return corrected_data