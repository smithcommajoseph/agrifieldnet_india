import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.file_ops import get_filepath

def get_field_paths(df):
    file_paths = (df['field_path']
        .value_counts()
        .rename_axis('field_path')
        .reset_index(name='pixel_count')
        ['field_path']
        .to_list()
    ) 
    return file_paths

def get_folder_ids(df):
    return (df['folder_id']
        .value_counts()
        .rename_axis('folder_id')
        .reset_index(name='pixel_count')
        ['folder_id']
        .to_list()
    ) 

#Extract field_crop Pairs 
def field_crop_extractor(crop_field_files, collection_id, train_label_collection):
    field_crops = {}

    for label_field_file in tqdm(crop_field_files):
        with rasterio.open(f'{collection_id}/{train_label_collection}/{train_label_collection}_{label_field_file}/field_ids.tif') as src:
            field_data = src.read()[0]
        with rasterio.open(f'{collection_id}/{train_label_collection}/{train_label_collection}_{label_field_file}/raster_labels.tif') as src:
            crop_data = src.read()[0]
    
        for x in range(0, crop_data.shape[0]):
            for y in range(0, crop_data.shape[1]):
                field_id = str(field_data[x][y])
                field_crop = crop_data[x][y]

                if field_crops.get(field_id) is None:
                    field_crops[field_id] = []

                if field_crop not in field_crops[field_id]:
                    field_crops[field_id].append(field_crop)
    
    field_crop_map  =[[k, v[0]]  for k, v in field_crops.items() ]
    field_crop = pd.DataFrame(field_crop_map , columns=['field_id','crop_id'])

    return field_crop[field_crop['field_id']!='0']


def pixel_data_extractor_1(data_, path, collection_id, selected_bands, n_obs, img_sh):
    '''
        data_: Dataframe with 'field_paths' and 'unique_folder_id' columns
        path: Path to source collections files

        returns: pixel dataframe with corresponding field_ids
        '''
    n_selected_bands = len(selected_bands)
    X = np.empty((0, n_selected_bands * n_obs))
    X_tile = np.empty((img_sh * img_sh, 0))
    X_arrays = []
        
    field_ids = np.empty((0, 1))
    folder_ids = np.empty((0, 1))
    lons = np.empty((0, 1))
    lats = np.empty((0, 1))

    for idx, tile_id in tqdm(enumerate(data_['unique_folder_id'])):
        f_path = data_['field_paths'].values[idx]

        field_src =   rasterio.open(f_path)
        field_array = field_src.read(1)
        field_ids = np.append(field_ids, field_array.flatten())

        cols, rows = np.meshgrid(np.arange(img_sh), np.arange(img_sh))
        xs, ys = rasterio.transform.xy(field_src.transform, rows, cols)   
        lons = np.append(lons, np.array(xs))
        lats = np.append(lats, np.array(ys))

        fids = [tile_id if fid != 0 else None for fid in field_array.flatten()]
        folder_ids = np.append(folder_ids, fids)
        
        bands_src = [rasterio.open(get_filepath(collection_id, path, tile_id, band)) for band in selected_bands]
        bands_array = [np.expand_dims(band.read(1).flatten(), axis=1) for band in bands_src]
        
        X_tile = np.hstack(bands_array)
        X_arrays.append(X_tile) 
        field_src.close()       

    X = np.concatenate(X_arrays)
    data = pd.DataFrame(X, columns=selected_bands)
    data['field_id'] = field_ids.astype(int)
    data['folder_id'] = folder_ids
    data['lon'] = lons
    data['lat'] = lats

    return data[data['field_id']!=0]
    
def rol_col_extractor(data_, path, collection_id, selected_bands):
    uni_field_keys = data_['merged_field_keys'].unique()
            
    field_ids = np.empty((0, 1))
    lons = np.empty((0, 1))
    lats = np.empty((0, 1))
    f_rows = np.empty((0, 1))
    f_cols = np.empty((0, 1))

    for field_key in tqdm(uni_field_keys):
        target_fields = data_[data_['merged_field_keys'] == field_key]['field_id'].unique()

        f_path = get_filepath(
            collection_id, path, field_key, selected_bands[0]
        )   
        
        field_src =   rasterio.open(f_path)
        field_array = field_src.read(1)
        temp_field_ids = np.array(field_array.flatten()).astype(int)

        # mask = field_ids in target_fields
        mask = np.in1d(temp_field_ids, target_fields, invert=True)
        masked_field_ids = np.ma.masked_where(mask, temp_field_ids)
        masked_field_ids = masked_field_ids.filled(fill_value=0)

        field_ids = np.append(field_ids, masked_field_ids)

        x_dim = field_src.width
        y_dim = field_src.height
        cols, rows = np.meshgrid(np.arange(x_dim), np.arange(y_dim))

        xs, ys = rasterio.transform.xy(field_src.transform, rows, cols)   
        lons = np.append(lons, np.array(xs))
        lats = np.append(lats, np.array(ys))
        f_cols = np.append(f_cols, np.array(cols))
        f_rows = np.append(f_rows, np.array(rows))
        field_src.close()    

    data = pd.DataFrame()
    data['field_id'] = field_ids.astype(int)
    data['lon'] = lons
    data['lat'] = lats
    data['row'] = f_rows.astype(int)
    data['col'] = f_cols.astype(int)

    return data[data['field_id']!=0]

def pixel_data_row_col_merger(train_data, row_col_data):
    merged_train_data = pd.merge(
        train_data, 
        row_col_data.drop(['field_id'], axis=1), 
        how='left', 
        on=['lat','lon'])

    update_cols = ['row','col']
    merged_train_data[update_cols] = merged_train_data[update_cols].applymap(np.int64)
    
    return merged_train_data

def aggregate_vals_by_field(group, field_id):
    agg_vals = {'field_id' : field_id}
    
    agg_vals['pixels'] = int(group.shape[0])
    
    for column_name, column in group.drop(['field_id', 'folder_id'],axis=1).iteritems():
        agg_vals[f'{column_name}_median'] = column.median()
        agg_vals[f'{column_name}_mean'] = column.mean()
        agg_vals[f'{column_name}_std'] = column.std()
        agg_vals[f'{column_name}_range'] = column.max() - column.min()
                   
    return agg_vals

def group_and_aggregate(df):
    agg_vals = []
    for field_id, group in tqdm(df.groupby('field_id', as_index=True)):
        
        #get field agg data 
        field_data = aggregate_vals_by_field(group, field_id)
        
        agg_vals.append(field_data)

    ret_val = pd.DataFrame.from_records(agg_vals)

    #fill nulls during std dev cal
    ret_val[ret_val == np.inf] = np.nan
    ret_val = ret_val.fillna(0)    

    return ret_val
