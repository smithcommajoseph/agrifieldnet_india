import subprocess
import numpy as np
import pandas as pd
from math import ceil
import rasterio
from osgeo import gdal
from rasterio.enums import Resampling
from os.path import exists
from utils.file_ops import *
from tqdm import tqdm

# borrowed from https://here.isnew.info/how-to-save-a-numpy-array-as-a-geotiff-file-using-gdal.html
def read_geotiff(filename):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr, band, ds

# borrowed from https://here.isnew.info/how-to-save-a-numpy-array-as-a-geotiff-file-using-gdal.html
def write_geotiff(filename, arr, in_ds):
    if arr.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    else:
        arr_type = gdal.GDT_Int32

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.FlushCache()
    band.ComputeStatistics(False)

def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

def mass_center(points):
    x_coordinates, y_coordinates = zip(*points)
    c_x = ceil((max(x_coordinates) + min(x_coordinates)) / 2)
    c_y = ceil((max(y_coordinates) + min(y_coordinates)) / 2)
    return (c_x, c_y)
    
def get_target_size_in_deg(bounding_box, padding=0):
    x_diff = bounding_box[1][0] - bounding_box[0][0]
    y_diff = bounding_box[1][1] - bounding_box[0][1]
    target_dim = max(x_diff, y_diff) + padding * 2
    return target_dim

def get_target_bounds(center, target_img_size):
    ul_x = round(center[0] - (target_img_size / 2))
    ul_y = round(center[1] - (target_img_size / 2))
    ul_x = ul_x if ul_x > 0 else 0
    ul_y = ul_y if ul_y > 0 else 0
    lr_x = round((center[0] + 1) + round(target_img_size / 2))
    lr_y = round((center[1] + 1) + (target_img_size / 2))
    return [(ul_x, ul_y), (lr_x, lr_y)]

def merge_tiles(merged_field_key, selected_bands, src_folders, collection_id, path):
    for band in selected_bands:
        target_vrt = get_filepath(
            collection_id, path, merged_field_key, band, 'vrt'
        )
        target = get_filepath(
            collection_id, path, merged_field_key, band
        )    
        create_dir_if_not_exists(target)
        if exists(target):
            print(f'{target} already exists')
        else:
            cmd = f'gdal_merge.py -ps 10 -10 -o {target}'
            subprocess.call(cmd.split() + src_folders[band])

            # build virtual raster and convert to geotiff
            vrt = gdal.BuildVRT(target_vrt, src_folders[band])
            gdal.Translate(target, vrt, xRes = 10, yRes = -10)
            vrt = None

def create_merged_tiles(df, source_collection_id, target_collection_id, 
                        label_collection, source_collection, selected_bands,
                        path):
    selected_names = ['field_ids']
    field_ids = df['field_id'].unique()
    
    field_ids_to_keys_df = pd.DataFrame()
    df_able_field_ids = []
    merged_field_keys = []

    for field_id in field_ids:
        field_df = df[df['field_id'] == field_id]

        folder_ids = get_folder_ids(field_df)
        # 
        folder_ids.sort()
        merged_field_key = '_'.join(folder_ids)

        if merged_field_key not in merged_field_keys:
            #labels
            src_folders_labels = get_file_source_folders(
                field_df, selected_names, source_collection_id, label_collection
            )     
            #bands
            src_folders_bands = get_file_source_folders(
                field_df, selected_bands, source_collection_id, source_collection
            ) 
            #labels
            merge_tiles(
                merged_field_key, selected_names, src_folders_labels, target_collection_id, path
            )
            #bands
            merge_tiles(
                merged_field_key, selected_bands, src_folders_bands, target_collection_id, path
            )  

        df_able_field_ids.append(field_id)
        merged_field_keys.append(merged_field_key)
            
    field_ids_to_keys_df['field_id'] = df_able_field_ids
    field_ids_to_keys_df['merged_field_keys'] = merged_field_keys

    return field_ids_to_keys_df

def get_merged_file_key(field_id, merge_map):
    return merge_map[merge_map['field_id'] == field_id]['merged_field_keys'].values[0]

def get_field_geo_info(df, padding):
    lon_lat_pairs = list(zip(df['row'], df['col']))

    bb = bounding_box(lon_lat_pairs)
    mc = mass_center(lon_lat_pairs)
    target_size = get_target_size_in_deg(bb, padding)
    target_bounds = get_target_bounds(mc, target_size)

    return bb, mc, target_size, target_bounds

def mask_and_save_image(merged_file_key, target_size, target_bounds, 
                        merged_band, merged_src, processed_collection_id, 
                        src_path, dest_path, field_id, band):

    src_tile = get_filepath(
        processed_collection_id, 
        src_path, 
        merged_file_key, 
        band)
        
    dest_tile = get_filepath(
        processed_collection_id, 
        dest_path, 
        field_id, 
        band)

    if not exists(dest_tile):
        create_dir_if_not_exists(dest_tile)     

        _, src_band, src = read_geotiff(src_tile)
        
        or_x = target_bounds[0][1]
        or_y = target_bounds[0][0]

        if (target_bounds[0][1] + target_size) > src.RasterXSize:
            or_x = src.RasterXSize - target_size
        if (target_bounds[0][0] + target_size) > src.RasterYSize:
            or_y = src.RasterYSize - target_size

        band_array = src_band.ReadAsArray(or_x, or_y, target_size, target_size)
        label_array = merged_band.ReadAsArray(or_x, or_y, target_size, target_size)
        mask = label_array != field_id

        new_x = np.ma.masked_where(mask, band_array)
        new_x = new_x.filled(fill_value=0)
        write_geotiff(dest_tile, new_x, merged_src)
        # 
        src = None
    return dest_tile

def rescale_and_save_image(processed_collection_id, dest_collection_id, src_path, dest_path, field_id, 
                           band, target_width=128, target_height=128, 
                           resampling=Resampling.nearest):
    
    src_tile = get_filepath(
        processed_collection_id, 
        src_path, 
        field_id, 
        band)

    dest_tile = get_filepath(
        dest_collection_id, 
        dest_path, 
        field_id, 
        band)

    if not exists(dest_tile):
        create_dir_if_not_exists(dest_tile)

        # open orig, masked image
        with rasterio.open(src_tile) as dataset:

            # resample data to target shape
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    target_width,
                    target_height
                ),
                # resampling=Resampling.average
                resampling=resampling
            )

            # scale image transform
            transform = dataset.transform * dataset.transform.scale(
                (dataset.width / data.shape[-1]),
                (dataset.height / data.shape[-2])
            )

            profile = dataset.profile
            profile.update(transform=transform, driver='GTiff',
                            height=data.shape[1], width=data.shape[2])

            with rasterio.open(dest_tile, 'w', **profile) as dst:
                            dst.write(data)

    return dest_tile 

def create_masked_and_scaled_field_files(merged_train_data, merge_map, 
                                         processed_collection_id, selected_bands,
                                         padding=0):
    # OPEN ONE SRC FILE AT A TIME HERE!
    merged_files = merge_map['merged_field_keys'].unique()

    for merged_file_key in tqdm(merged_files):
        field_ids = merge_map[merge_map['merged_field_keys'] == merged_file_key]['field_id'].values
        merged_tile = get_filepath(
            processed_collection_id, 
            'merged_TRAIN_v2',
            merged_file_key, 
            'field_ids')

        _, merged_band, merged_src = read_geotiff(merged_tile)
        
        # FOR EACH FIELD_ID
        for field_id in field_ids:
            field_data = merged_train_data[merged_train_data['field_id'] == field_id]
            _, _, target_size, target_bounds = get_field_geo_info(field_data, padding)        
                        
            ## FOR EACH SELECTED_BAND
            for band in selected_bands:
                masked_img_path = mask_and_save_image(
                    merged_file_key,
                    target_size,
                    target_bounds,
                    merged_band,
                    merged_src,
                    processed_collection_id, 
                    'merged_TRAIN_v2',
                    'masked_TRAIN',
                    field_id, 
                    band)

                scaled_img_path = rescale_and_save_image(
                    processed_collection_id, 
                    'data',
                    'masked_TRAIN',
                    'scaled_TRAIN', 
                    field_id, 
                    band)
        # 
        merged_src = None    
    