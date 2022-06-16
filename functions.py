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

def normalize_arr(X_data) -> pd.DataFrame:
    arr = []
    for col in X_data.T:
        col_arr = np.array(list(col), dtype=np.float64)
        
        x_mean = np.mean(col_arr, axis=(0,1))
        x_var = np.var(col_arr, axis=(0,1))
        col_arr -= x_mean
        if x_var:
            col_arr /= x_var
        else:
            col_arr *= 0
        
        arr.append(col_arr)
        
        
    arr = np.transpose(np.array(arr, dtype=object), [1,0,2])
    return arr


def normalize_dif_len(X_data) -> pd.DataFrame: # Исправить
    arr = []
    got_lenghts = False
    for col in X_data.T:
        
        colflat = [] 
        if not got_lenghts:
            lengths = []
        for traj_arr in col:
            colflat += list(traj_arr)
            if not got_lenghts:
                lengths.append(len(traj_arr))
        
        if not got_lenghts:
            got_lenghts = True
            
        col_arr = np.array(colflat, dtype=np.float64)
        
        x_mean = np.mean(col_arr)
        x_var = np.var(col_arr)
        col_arr -= x_mean
        if x_var:
            col_arr /= x_var
        else:
            col_arr *= 0
        
        col_arr_reshaped = []
        start = 0
        for i in range(len(lengths)):
            traj_arr = col_arr[start:start+lengths[i]]
            start += lengths[i]
            col_arr_reshaped.append(traj_arr)
            
        arr.append(col_arr_reshaped)
        
        
    return np.array(arr, dtype=object).T



def cut_act(df, cut_len, count=-1, random_start = False) -> pd.DataFrame:
    '''
    Consideres that cut_len is le to the lenght of all activities
    count is maximum number of cutted signals from one activity (-1 is default for maximum number)
    random start works with due regard for count
    '''
    tdf = pd.DataFrame(columns=good_cols, dtype = object)
    for i in tqdm(range(len(df))):
        l = len(df.iloc[i]['heart rate'])
        start = np.random.randint(0, l-count*(l//count)+1) if random_start and count !=-1 else 0
        el = np.array(df.iloc[i], dtype = object)
        for j in range(start, l-cut_len, cut_len):
            if j//cut_len == count:
                break
            new_el = [el[0]]
            for k in range(1, len(el)):
                new_el.append(el[k][j:j+cut_len])
            tdf = tdf.append(pd.DataFrame([new_el], columns=good_cols))
    
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

class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, data_X, data_Y):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = data_X
        self.Y = data_Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        input_data = self.X[idx]
        label = self.Y[idx]
        
        return input_data, label

def get_equal_len(X, need_len=-1, fill_with=0): # Доделать
    '''
    By default sets all signals to len of signal with maximum length, putting fill_with to the end of signal
    If you know precise len you need, it can be changed, signals with bigger lenght would be cutted
    '''
    if need_len == -1:
        max_len = -1
        for sig_arr in X.T[0]:
            max_len = max(max_len, len(sig_arr))
        
        X_eq = []    
        for sig in X:
            sig_arr = np.array(list(sig), dtype=np.float64)
#             print(sig_arr.shape, max_len)
            z = np.zeros((sig_arr.shape[0], max_len-sig_arr.shape[1]), dtype=np.float64)
            new_arr = np.concatenate((sig_arr,z), axis=1)
#             print(new_arr.shape, max_len)
            X_eq.append(new_arr)
        return np.array(X_eq, dtype = np.float64)
        
    else: # Пока не сделано
        pass
    


    
    
