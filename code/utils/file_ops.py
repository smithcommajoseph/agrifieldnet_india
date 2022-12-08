from yaml import load
import pandas as pd
from os.path import exists
from os import makedirs

def get_filepath(collection_id, path, tile_id, band, ext='tif'):
   return f'{collection_id}/{path}/{path}_{tile_id}/{band}.{ext}' 

def create_dir_if_not_exists(file_name):
    sp = file_name.split('/')
    path = '/'.join(sp[0:len(sp)-1])
    if not exists(path):
        makedirs(path)

def get_folder_ids(df):
    return (df['folder_id']
        .value_counts()
        .rename_axis('folder_id')
        .reset_index(name='pixel_count')
        ['folder_id']
        .to_list()
    ) 

def get_file_source_folders(df, selected_bands, collection_id, source_collection):
    folder_ids = get_folder_ids(df)
    src_folders = {}

    for folder_id in folder_ids:
        for band in selected_bands:
            src = get_filepath(
                collection_id, source_collection, folder_id, band
            ) 
            if not src_folders.get(band):
                src_folders[band] = []
            src_folders[band].append(src)
    
    return src_folders    

def get_df_from_csv_if_exists(file_name, false_cb):
    if exists(file_name):
       df = pd.read_csv(file_name, index_col=[0])
    else:
        df = false_cb()

    return df
    
# 
def write_csv_from_df(df, out_file_name, write_compressed_ver=True):
    if exists(f'{out_file_name}.csv'):
        print(f'the file {out_file_name}.csv already exists.')
    elif exists(f'{out_file_name}.zip'):
        print(f'the file {out_file_name}.zip already exists.')   
    else: 
        # csv
        df.to_csv(f'{out_file_name}.csv')
        # compressed
        if write_compressed_ver:
            df.to_csv(f'{out_file_name}.zip')

# 
def read_yaml(file_path):
    try:
        from yaml import CLoader as Loader, CDumper as Dumper
    except ImportError:
        from yaml import Loader, Dumper
    with open(file_path, 'r') as f:
        return load(f, Loader=Loader)