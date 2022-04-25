import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from skimage import transform
import torch


good_cols = ['activityID', 'heart rate', 'temperature hand',\
             '3Da_x scale_16 hand', '3Da_y scale_16 hand', '3Da_z scale_16 hand', \
             '3Dg_x hand', '3Dg_y hand', '3Dg_z hand', '3Dm_x hand', '3Dm_y hand', '3Dm_z hand', \
             'temperature chest', '3Da_x scale_16 chest', '3Da_y scale_16 chest', '3Da_z scale_16 chest', \
             '3Dg_x chest', '3Dg_y chest', '3Dg_z chest', '3Dm_x chest', '3Dm_y chest', '3Dm_z chest', \
             'temperature ankle', '3Da_x scale_16 ankle', '3Da_y scale_16 ankle', '3Da_z scale_16 ankle', \
             '3Dg_x ankle', '3Dg_y ankle', '3Dg_z ankle', '3Dm_x ankle', '3Dm_y ankle', '3Dm_z ankle']

def y_encode(y_data):
    y_targ = np.zeros_like(y_data)
    code = list(np.unique(y_data))
    for i, el in enumerate(y_data):
        y_targ[i] = code.index(el)
        
    return np.array(y_targ, dtype = np.int32)

def generate_batches(X, y, batch_size=64):
    for i in range(0, X.shape[0], batch_size):
        X_batch, y_batch = X[i:i+batch_size], y[i:i+batch_size]
        yield X_batch, y_batch

def normalize_array(arr) -> np.ndarray:
    x_min = min(arr)
    x_max = max(arr)
    if x_min != x_max:
        return (arr-x_min)/(x_max-x_min)
    else:
        return 0*arr
    

def normalize_df(df) -> pd.DataFrame: # Доделать
    arr = []
    for i in range(len(df)):
        arr.append([df.iloc[i]['activityID']])
        for el in df.iloc[i][1:]:
            arr[i].append(normalize_array(el))
    
#     print(arr)
    return pd.DataFrame(arr, columns=good_cols, dtype = object)


def cut_act(df, cut_len, count=-1, random_start = False) -> pd.DataFrame:
    '''
    Consideres that cut_len is le to the lenght of all activities
    count is maximum number of cutted signals from one activity (-1 is default for maximum number)
    random start works with due regard for count
    '''
    tdf = pd.DataFrame(columns=good_cols, dtype = object)
    for i in tqdm(range(len(df))):
        l = len(df.iloc[i]['heart rate'])
        start = np.random.randint(0, l-count*(l//count)+1) if random_start else 0
        el = np.array(df.iloc[i], dtype = object)
#         print(el[1], '\n')
        for j in range(start, l-cut_len, cut_len):
            if j//cut_len == count:
                break
            new_el = [el[0]]
            for k in range(1, len(el)):
                new_el.append(el[k][j:j+cut_len])
            tdf = tdf.append(pd.DataFrame([new_el], columns=good_cols))
#             pd.DataFrame.append()
#             print(tdf, '\n')
    
    tdf.index = pd.Int64Index(list(range(len(tdf))))
    return tdf


def sep_by_len(df, min_act_len, ret_min_len = False) -> (pd.DataFrame, np.ndarray):
    uniq_act = df['activityID'].unique().tolist()
    min_len = np.zeros((len(uniq_act)), dtype = np.int64) - 1
    chosen = []

    for i in range(len(df)):
        p = df.iloc[i]
        if len(p['heart rate']) >= min_act_len:
            chosen.append(i)
            if min_len[uniq_act.index(p['activityID'])] == -1:
                min_len[uniq_act.index(p['activityID'])] = len(p['heart rate'])
            else:
                min_len[uniq_act.index(p['activityID'])] = min(min_len[uniq_act.index(p['activityID'])], len(p['heart rate']))
    
    if ret_min_len:
        return min_len
    else:
        return df.iloc[(chosen)] 

def get_good_data(fname, delete_zero_activity = True):    
    col_names = ['timestamp', 'activityID', 'heart rate', 'temperature hand',\
             '3Da_x scale_16 hand', '3Da_y scale_16 hand', '3Da_z scale_16 hand', \
             '3Da_x scale_6 hand', '3Da_y scale_6 hand', '3Da_z scale_6 hand', \
             '3Dg_x hand', '3Dg_y hand', '3Dg_z hand', '3Dm_x hand', '3Dm_y hand', '3Dm_z hand', \
             'orientation_0 hand', 'orientation_1 hand', 'orientation_2 hand', 'orientation_3 hand', 
             'temperature chest', '3Da_x scale_16 chest', '3Da_y scale_16 chest', '3Da_z scale_16 chest', \
             '3Da_x scale_6 chest', '3Da_y scale_6 chest', '3Da_z scale_6 chest', \
             '3Dg_x chest', '3Dg_y chest', '3Dg_z chest', '3Dm_x chest', '3Dm_y chest', '3Dm_z chest', \
             'orientation_0 chest', 'orientation_1 chest', 'orientation_2 chest', 'orientation_3 chest',
             'temperature ankle', '3Da_x scale_16 ankle', '3Da_y scale_16 ankle', '3Da_z scale_16 ankle', \
             '3Da_x scale_6 ankle', '3Da_y scale_6 ankle', '3Da_z scale_6 ankle', \
             '3Dg_x ankle', '3Dg_y ankle', '3Dg_z ankle', '3Dm_x ankle', '3Dm_y ankle', '3Dm_z ankle', \
             'orientation_0 ankle', 'orientation_1 ankle', 'orientation_2 ankle', 'orientation_3 ankle']
    
    data = pd.read_csv(fname, names = col_names, sep = ' ')
    data_gc = data[good_cols]
    if delete_zero_activity:
        data_gc = data_gc[(data_gc.activityID != 0)]
        
    return data_gc
 
def data_fill_na(data):
    data_nonans = data.interpolate(axis = 0, method='linear')
    data_nonans = data_nonans.fillna(axis = 0, method='bfill')
    data_nonans = data_nonans.fillna(axis = 0, method='ffill')
    return data_nonans

def get_activity(data, activityID, with_fill = True):
    data_act = data[(data.activityID == activityID)]
    data_act.pop('activityID')
    if with_fill:
        data_act = data_fill_na(data_act)
    return np.array(data_act).T

def get_df(): # Добавить выбор файлов для загрузки
    adf = pd.DataFrame(columns=good_cols, dtype = object)

    for i in tqdm(range(1,10)):
        subj_fname = f'PAMAP2_Dataset/Protocol/subject10{i}.dat'
        subj_df = get_good_data(subj_fname)
        df_arr = []
        uniq_act = subj_df['activityID'].unique().tolist()
        for act in uniq_act:
            arr = []
            arr.append(act)
            arr += list(get_activity(subj_df, act))
            if len(arr[1]):
                df_arr.append(arr)

        add = pd.DataFrame(data = np.array(df_arr, dtype = object), dtype = object, columns=good_cols)
        adf = pd.concat([adf,add])
    
    adf.index = pd.Int64Index(list(range(len(adf))))
    return adf


def get_flatten(data):
    flat_data = []
    for el in data:
        flat = []
        for col in el:
            flat += list(col)
        flat_data.append(flat)
        
    return np.array(flat_data)




