
"""
preprocessing.py

"""

import ee
import numpy as np


# GEE Functions (run on Google servers)

def mask_clouds_scl(image):
    """
    Remove clouds, cloud shadows, and low-quality pixels from a
    Sentinel-2 Level 2A image using the Scene Classification Layer (SCL).
    Keeps: SCL 4 (vegetation), 5 (bare soil), 6 (water), 11 (snow).
    Removes: clouds (8,9,10), shadows (3), saturated (1,2), unclassified (7).
    """
    scl = image.select('SCL')
    valid_mask = (scl.eq(4)
                 .Or(scl.eq(5))
                 .Or(scl.eq(6))
                 .Or(scl.eq(11)))
    return image.updateMask(valid_mask)


def add_spectral_indices(image):
    """
    Add NDVI, NDWI, NBR, EVI as new bands to a Sentinel-2 image.
    Must be called AFTER mask_clouds_scl.
    """
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    nbr  = image.normalizedDifference(['B8', 'B12']).rename('NBR')
    evi  = image.expression(
        '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
        {'NIR': image.select('B8'), 'RED': image.select('B4'),
         'BLUE': image.select('B2')}
    ).rename('EVI')
    return image.addBands([ndvi, ndwi, nbr, evi])


def make_monthly_composite(year, month, collection, bands):
    """
    Create a monthly median composite image.
    collection must already have cloud masking and indices applied.
    """
    start = ee.Date.fromYMD(year, month, 1)
    end   = start.advance(1, 'month')
    return (collection
        .filterDate(start, end)
        .select(bands)
        .median()
        .set('year', year)
        .set('month', month)
        .set('system:time_start', start.millis()))


# NumPy Functions (run locally after export)

def interpolate_time_series(X):
    """
    Fill NaN gaps in pixel time series using linear interpolation.
    X shape: (n_pixels, n_timesteps, n_features)
    Returns: X with NaN filled, count of interpolated values.
    """
    X_out = X.copy()
    n_interp = 0
    t = np.arange(X.shape[1])
    for i in range(X.shape[0]):
        for f in range(X.shape[2]):
            ts = X_out[i, :, f]
            nan_mask = np.isnan(ts)
            if nan_mask.any() and (~nan_mask).sum() >= 2:
                ts[nan_mask] = np.interp(t[nan_mask], t[~nan_mask], ts[~nan_mask])
                n_interp += nan_mask.sum()
                X_out[i, :, f] = ts
    return X_out, n_interp


def compute_spectral_indices_numpy(X, band_names):
    """
    Compute NDVI, NDWI, NBR, EVI from raw band arrays and append.
    X shape: (n_pixels, n_timesteps, n_raw_bands)
    Returns: (n_pixels, n_timesteps, n_raw_bands + 4)
    """
    eps = 1e-10
    b2  = X[:, :, band_names.index('B2')]
    b3  = X[:, :, band_names.index('B3')]
    b4  = X[:, :, band_names.index('B4')]
    b8  = X[:, :, band_names.index('B8')]
    b12 = X[:, :, band_names.index('B12')]
    ndvi = np.clip((b8 - b4)  / (b8 + b4  + eps), -1, 1)
    ndwi = np.clip((b3 - b8)  / (b3 + b8  + eps), -1, 1)
    nbr  = np.clip((b8 - b12) / (b8 + b12 + eps), -1, 1)
    evi  = np.clip(2.5 * (b8 - b4) / (b8 + 6*b4 - 7.5*b2 + 1 + eps), -1, 1)
    return np.concatenate([X, np.stack([ndvi, ndwi, nbr, evi], axis=-1)], axis=-1).astype(np.float32)
