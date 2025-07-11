import numpy as np
from scipy.ndimage import gaussian_filter1d


def filter_gaussian(x_vals, y_vals, sigma=1.0):
    """Filter y values using a Gaussian filter for smoothing.
    
    Args:
        x_vals: Array of x coordinates
        y_vals: Array of y values to be filtered
        sigma: Standard deviation for Gaussian kernel (controls smoothing amount)
    
    Returns:
        Filtered y values array
    """
    valid_mask = ~np.isnan(y_vals) & (y_vals > 0)
    y_valid = y_vals.copy()
    
    if np.any(valid_mask):
        # Interpolate NaN values with nearest valid neighbors
        y_valid[~valid_mask] = np.interp(
            x_vals[~valid_mask],
            x_vals[valid_mask],
            y_vals[valid_mask]
        )
        
        # Apply Gaussian filter
        filtered_values = gaussian_filter1d(y_valid, sigma=sigma)
        
        # Preserve original zeros and NaNs
        filtered_values[~valid_mask] = y_vals[~valid_mask]
    else:
        filtered_values = y_vals.copy()
    
    return filtered_values