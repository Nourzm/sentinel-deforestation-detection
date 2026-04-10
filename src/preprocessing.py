"""
Reusable functions for Sentinel-2 data preprocessing.
"""
import ee
import numpy as np


def mask_clouds_scl(image):
    """
    Remove clouds, cloud shadows, and low-quality pixels from a
    Sentinel-2 Level 2A image using the Scene Classification Layer (SCL).
    
    Keeps only:
        SCL = 4  (vegetation)
        SCL = 5  (bare soil / non-vegetated)
        SCL = 6  (water)
        SCL = 11 (snow/ice — rare in tropical study areas)
    
    All other pixels (clouds, shadows, saturated, cirrus) are masked
    and treated as missing data in downstream compositing.
    
    Parameters
    ----------
    image : ee.Image
        A single Sentinel-2 SR image from the GEE collection.
    
    Returns
    -------
    ee.Image
        The same image with invalid pixels masked out.
    """
    scl = image.select('SCL')
    
    # Build a mask: 1 where pixel is valid, 0 where it should be removed
    valid_mask = (scl.eq(4)    # vegetation — primary target
                 .Or(scl.eq(5))  # bare soil, urban
                 .Or(scl.eq(6))  # water bodies
                 .Or(scl.eq(11))) # snow/ice
    
    # Apply the mask: masked pixels will appear as no-data
    return image.updateMask(valid_mask)


def add_spectral_indices(image):
    """
    Add computed spectral indices as new bands to a Sentinel-2 image.
    
    Indices added:
        NDVI  = (B8 - B4) / (B8 + B4)   — vegetation health
        NDWI  = (B3 - B8) / (B3 + B8)   — water content
        NBR   = (B8 - B12) / (B8 + B12) — burn ratio
        EVI   = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1) — enhanced vegetation
    
    Parameters
    ----------
    image : ee.Image
        A Sentinel-2 image with bands B2, B3, B4, B8, B12.
    
    Returns
    -------
    ee.Image
        The same image with four additional bands: NDVI, NDWI, NBR, EVI.
    """
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    nbr  = image.normalizedDifference(['B8', 'B12']).rename('NBR')
    
    # EVI requires a more complex formula — use expression()
    evi = image.expression(
        '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
        {
            'NIR':  image.select('B8'),
            'RED':  image.select('B4'),
            'BLUE': image.select('B2')
        }
    ).rename('EVI')
    
    return image.addBands([ndvi, ndwi, nbr, evi]) 