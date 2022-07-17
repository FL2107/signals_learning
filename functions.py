import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from skimage import transform
import torch
from typing import Union 
from sklearn.model_selection import train_test_split
import torch.nn as nn
import os


good_cols = ['activityID', 'heart rate', 'temperature hand',\
             '3Da_x scale_16 hand', '3Da_y scale_16 hand', '3Da_z scale_16 hand', \
             '3Dg_x hand', '3Dg_y hand', '3Dg_z hand', '3Dm_x hand', '3Dm_y hand', '3Dm_z hand', \
             'temperature chest', '3Da_x scale_16 chest', '3Da_y scale_16 chest', '3Da_z scale_16 chest', \
             '3Dg_x chest', '3Dg_y chest', '3Dg_z chest', '3Dm_x chest', '3Dm_y chest', '3Dm_z chest', \
             'temperature ankle', '3Da_x scale_16 ankle', '3Da_y scale_16 ankle', '3Da_z scale_16 ankle', \
             '3Dg_x ankle', '3Dg_y ankle', '3Dg_z ankle', '3Dm_x ankle', '3Dm_y ankle', '3Dm_z ankle']


def print_signal(signal: Union[np.ndarray, list], style='plot') -> None:
    '''
    Draws plot or scatter (set by style) for signal or list of signals
    style: dot or plot
    '''
    if isinstance(signal, list) or (isinstance(signal, np.ndarray) and signal[0].size != 1):
        N = len(signal)
        fig, axs = plt.subplots(N,1,figsize=(14,8*N))
        for i, sig in enumerate(signal):
            x = np.arange(sig.shape[0])
            if style == 'plot':
                axs[i].plot(x, sig)
            elif style == 'dot':
                axs[i].scatter(x, sig, s = 14*120/sig.shape[0])
        
    elif isinstance(signal, np.ndarray):
        fig, axs = plt.subplots(1,1,figsize=(14,8))
        x = np.arange(signal.shape[0])
        if style == 'plot':
            axs.plot(x, signal)
        elif style == 'dot':
            axs.scatter(x, signal, s = 14*120/signal.shape[0])
    
    plt.show()

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
        self.X = data_X
        self.Y = data_Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        input_data = self.X[idx]
        label = self.Y[idx]
        
        return input_data, label
    
class SignalDataset_1param(torch.utils.data.Dataset):
    def __init__(self, data_X):
        self.X = data_X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        input_data = self.X[idx]
        
        return input_data

class Linear_Autoencoder(nn.Module):
    def __get_enc_layers(self, layers_param_count, activation):
        layers = []
        for i in range(len(layers_param_count)-2):
            layers.append(nn.Linear(layers_param_count[i], layers_param_count[i+1]))
            layers.append(activation())
        layers.append(nn.Linear(layers_param_count[-2], layers_param_count[-1]))
        return layers
    
    def __get_dec_layers(self, layers_param_count, activation):
        layers = []
        for i in range(len(layers_param_count)-1, 1, -1):
            layers.append(nn.Linear(layers_param_count[i], layers_param_count[i-1]))
            layers.append(activation())
        layers.append(nn.Linear(layers_param_count[1], layers_param_count[0]))
        return layers
    
    def __init__(self, layers_param_count, activation):
        super(Linear_Autoencoder, self).__init__()
        layers_enc = self.__get_enc_layers(layers_param_count, activation)
        self.encoder = nn.Sequential(*layers_enc)
        
        layers_dec = self.__get_dec_layers(layers_param_count, activation)
        self.decoder = nn.Sequential(*layers_dec)
    
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec

    
    
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

def get_mse_delta(y_true: torch.Tensor, y_net: torch.Tensor) -> float:
    loss = torch.nn.MSELoss()
    delta = loss(y_true, y_net).item()
    return delta

def MSE(x: np.ndarray, y: np.ndarray) -> float :
    return ((x-y)**2).mean()

def NMSE(x: np.ndarray, y: np.ndarray) -> float :
    a = MSE(x,y)
    b = MSE(x, np.zeros_like(x))
    if b != 0:
        return a/b
    elif a==0 and b==0:
        return 0
    else:
        return 1e8

def basic_func_check(model, sig_len, basics=6, sin_periods = 6):
    x = np.arange(sig_len)
    b0 = 2*np.sin(x*sin_periods/sig_len*2*np.pi)
    b1 = 20*np.sin(x*sin_periods/sig_len*2*np.pi)
    b2 = 2*np.ones((sig_len))
    b3 = 5*np.sqrt(x/10)
    b4 = np.arange(sig_len)/4
    b5 = 10+np.arange(sig_len)/2
    
    
    fig, axs = plt.subplots(basics,1,figsize=(14,8*basics))
    
    for i in range(basics):
#         exec("global old_string; old_string = new_string")
        exec(f'global base; base = b{i}')
        col_arr = np.array(list(base), dtype=np.float64)
        x_mean = np.mean(col_arr, axis=0)
        x_var = np.var(col_arr, axis=0)
        col_arr -= x_mean
        if x_var:
            col_arr /= x_var
        else:
            col_arr *= 0
        X_norm = col_arr
        
        out_normalized = model(torch.Tensor(X_norm)).detach().numpy()
        out = out_normalized * x_var
        out += x_mean
        
        axs[i].plot(x, base, color = 'black', label='Изначальный сигнал')
        axs[i].plot(x, out, color = 'red', label='Сгенерированный сигнал')
        axs[i].legend()
        delta = get_mse_delta(torch.Tensor(base),torch.Tensor(out))
        nmse = NMSE(base, out)
        axs[i].set(title=f"MSE: {np.round(delta,4)}\nNMSE: {np.round(nmse,5)}")
    
    plt.show()
    
def tensor_check(model, tensor, number_of_samples, mean, var, random_state=42, without_unnorm=False, without_sampling=False):
    np.random.seed(random_state)
    origin_normalized = tensor.detach().numpy()
    N = number_of_samples
    samples_idxs = np.arange(len(tensor)) if without_sampling else np.random.randint(0, len(tensor),size=N)
    fig, axs = plt.subplots(N,1,figsize=(14,8*N))
    x = np.arange(len(origin_normalized[0]))
    if without_unnorm: # undone
        for i in range(N):
            idx = samples_idxs[i]
            out_normalized = model(tensor[idx]).detach().numpy()
            axs[i].plot(x, origin_normalized[idx], color = 'black', label='Изначальный сигнал,\nнормализованный')
            axs[i].plot(x, out_normalized, color = 'red', label='Сгенерированный сигнал,\nнормализованный')
            axs[i].legend()
            delta = get_mse_delta(torch.Tensor(origin_normalized[idx]),torch.Tensor(out_normalized))
            nmse = NMSE(origin_normalized[idx], out_normalized)
            axs[i].set(title=f"number of sample: {idx}\nMSE: {np.round(delta,4)}\nNMSE: {np.round(nmse,5)}")
    
    else:
        origin = origin_normalized * var
        origin += mean
        
        for i in range(N):
            idx = samples_idxs[i]
            out_normalized = model(tensor[idx]).detach().numpy()
            out = out_normalized * var
            out += mean
            axs[i].plot(x, origin[idx], color = 'black', label='Изначальный сигнал')
            axs[i].plot(x, out, color = 'red', label='Сгенерированный сигнал')
            axs[i].legend()
            delta = get_mse_delta(torch.Tensor(origin[idx]),torch.Tensor(out))
            nmse = NMSE(origin[idx], out)
            axs[i].set(title=f"number of sample: {idx}\nMSE: {np.round(delta,4)}\nNMSE: {np.round(nmse,5)}")
        
    plt.show()

def get_tensors_1param(dataframe, target_len, cut_len, count_per_signal, needed_param, random_state=42, test_size=0.02, random_start=True, get_mean=True) -> tuple:
    activities = sep_by_len(dataframe, target_len)
    cut_df = cut_act(activities, cut_len, count=count_per_signal, random_start=random_start)
    X = cut_df.loc[:, needed_param].values
    Y = cut_df.iloc[:, 0] # целевая переменная
    y_targ = y_encode(Y)
    
    col_arr = np.array(list(X), dtype=np.float64)
    x_mean = np.mean(col_arr, axis=(0,1))
    x_var = np.var(col_arr, axis=(0,1))
    col_arr -= x_mean
    if x_var:
        col_arr /= x_var
    else:
        col_arr *= 0
    X_norm = col_arr
    
    X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = \
    train_test_split(torch.FloatTensor(X_norm), torch.LongTensor(y_targ), random_state=random_state, test_size = test_size)
    if get_mean:
        return X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, x_mean, x_var
    else:
        return X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor

def write_signals_1param(dataframe, sig_len, count_per_signal, needed_params) -> None:
    target_len = sig_len*count_per_signal

    activities = sep_by_len(dataframe, target_len)
    cut_df = cut_act(activities, sig_len, count_per_signal, random_start=False)
    for needed_param in needed_params:
        X = cut_df.loc[:, needed_param].values
        X = np.array(list(X), dtype=np.float64)

        if not os.path.isdir(f'1param_df/{needed_param}'):
            os.mkdir(f'1param_df/{needed_param}')
        np.save(f'1param_df/{needed_param}/sig_len{sig_len}', X)
    return
    
def load_signals_1param(sig_len, needed_param) -> Union[None, np.ndarray]:
    '''
    load np.ndaaray with signals of needed param and len in case of their existing
    '''
    if os.path.isdir(f'1param_df/{needed_param}'):
        if f'sig_len{sig_len}.npy' in os.listdir(f'1param_df/{needed_param}'):
            signals = np.load(f'1param_df/{needed_param}/sig_len{sig_len}.npy')
            return signals
        
    return  

def get_signal_groups(signals, nog, N=1000) -> list:
    '''
    nog = number of groups at the end
    N = number of groups in the initial partition (accuracy)
    '''
    def concat(arr, idx1, idx2):
        '''
        inplace operation!
        '''
        if idx1 < idx2:
            arr[idx1] += arr[idx2]
            arr.pop(idx2)
        elif idx1 > idx2:
            arr[idx2] += arr[idx1]
            arr.pop(idx1)
            
    sig_vars = signals.var(axis=1)
    sig_groups = [[] for _ in range(N)]
    MAX_VAR = np.max(sig_vars) + 1e-8
    PART_VAR = MAX_VAR/N
    for i, el in enumerate(sig_vars):
    #     print(int(el//PART_VAR))
        sig_groups[int(el//PART_VAR)].append(signals[i])

    group_lens = [len(el) for el in sig_groups]
    while len(sig_groups) != nog:
        min_idx = np.argmin(group_lens)
        lsg = len(sig_groups)
        if min_idx == 0:
            concat(sig_groups, 0, 1)
            concat(group_lens, 0, 1)
        elif min_idx == lsg-1:
            concat(sig_groups, lsg-2, lsg-1)
            concat(group_lens, lsg-2, lsg-1)
        else:
            if group_lens[min_idx-1] < group_lens[min_idx+1]:
                concat(sig_groups, min_idx, min_idx-1)
                concat(group_lens, min_idx, min_idx-1) 
            else:
                concat(sig_groups, min_idx, min_idx+1)
                concat(group_lens, min_idx, min_idx+1) 
                
    return sig_groups
        
 

