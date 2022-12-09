# NDVI SRI RENDVI ARI - https://journals-crea.4science.it/index.php/asr/article/view/1463
# other VIs - https://sentinel-hub.com/develop/documentation/eo_products/Sentinel2EOproducts
# Cleanly indexed version found here - https://www.indexdatabase.de/db/i.php

# Enhanced Vegetation Index (EVI)
def EVI(df):
    NIR = df['B08']
    R = df['B04']
    B = df['B02']
    GA = 2.5 
    C1 = 6
    C2 = 7.5
    L = 1
    return GA * ((NIR - R) / (NIR + C1 * R - C2 * B + L))

# Normalized Difference Vegetation Index (NDVI)
def NDVI(df):
    NIR = df['B08']
    R = df['B04']
    return (df['B08'] - df['B04']) / (df['B08'] + df['B04'])

# Green Normalized Difference Vegetation Index (GNDVI)
def GNDVI(df):
    NIR = df['B08']
    G = df['B03']
    NDVI = df['NDVI']

    return (NIR - G) / (NDVI + G)    

#Atmospherically Resistant Vegetation Index (ARVI)     
def ARVI(df):
    NIR = df['B08']
    R = df['B04']
    B = df['B02']
    GA = 2
    RB = (R - GA * (B - R))
    return (NIR - RB) / (NIR + RB)

# Soil-Adjusted Vegetation Index (SAVI)
def SAVI(df):
    NIR = df['B08']
    R = df['B04']
    GA = 1
    L = 0.5
    return (NIR - R) * (GA + L) / (NIR + R + L)

# Solar Reflectance Index (SRI)
def SRI(df):
    NIR = df['B08']
    R = df['B04']    
    return NIR / R

# Red Edge Normalized Difference Vegetation Index (RENDVI)
def RENDVI(df):
    VRE = df['B05']
    VRE2 = df['B06']
    return (VRE2 - VRE) / (VRE2 + VRE)

#  Anthocyanin Reflectance Index (ARI)
def ARI(df):
    G = df['B03']
    VRE = df['B05']
    return (1 / G) - (1 / VRE)

# Soil Adjusted Vegetation Index (SAVI)
def SAVI(df):
    NIR = df['B08']
    R = df['B04']    
    L = 0.428
    return (NIR - R) / (NIR + R + L) * (1.0 + L)

# Moisture Stress Index (MSI)
def MSI(df):
    NIR = df['B08']
    SWIR = df['B11']
    return SWIR / NIR

# Modified Chlorophyll Absorption in Reflective Index (MCARI)
def MCARI(df):
    R = df['B04']        
    G = df['B03']
    VRE = df['B05']
    return ((VRE - R) - 0.2 * (VRE - G)) * (VRE / R)

# Modified anthocyanin reflectance index (MARI)
def MARI(df):
    G = df['B03']
    VRE = df['B05']
    VRE3 = df['B07']
    return ((1.0 / G) - (1.0 / VRE)) * VRE3

# Enhanced Vegetation Index 2 (EVI2)
def EVI2(df):
    NIR = df['B08']
    R = df['B04']    
    return 2.4 * (NIR - R) / (NIR + R + 1.0)

# Normalized Difference 820/1600 Normalized Difference Moisture Index (NDMI)
def NDMI(df):
    NIR = df['B08']
    SWIR = df['B11']
    return (NIR - SWIR) / (NIR + SWIR)

# Normalized Difference 860/1240 Normalized Difference Water Index (NDWI)
def NDWI(df):
    NIR = df['B08']
    G = df['B03']
    return (G - NIR) / (G + NIR)


def add_vegetation_indices(df, selected_bands):
    # df['EVI'] = EVI(df)
    # df['GNDVI'] = GNDVI(df)
    df['NDVI'] = NDVI(df)
    df['ARVI'] = ARVI(df)
    df['SAVI'] = SAVI(df)
    df['SRI'] = SRI(df)
    df['RENDVI'] = RENDVI(df)
    df['ARI'] = ARI(df)
    df['SAVI'] = SAVI(df)
    df['MSI'] = MSI(df)
    df['MCARI'] = MCARI(df)
    df['MARI'] = MARI(df)
    df['EVI2'] = EVI2(df)
    df['NDMI'] = NDMI(df)
    df['NDWI'] = NDWI(df)

    df['brightness'] = df[selected_bands].apply(lambda x: (sum(abs(x)) / 10),axis=1)
