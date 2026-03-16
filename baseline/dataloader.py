import sys
sys.path.append("/mnt/data1/tyl/UserID/")
from re import T
from typing import Optional, List
from scipy import signal
from scipy.signal import resample
import scipy.io as scio
import numpy as np
from utils.data_alignment import centroid_align
from sklearn.model_selection import train_test_split
from utils.preprocess import preprocessing
from dataset.instance import trans_torch
import os
import mne
import scipy.io as scio
import matplotlib.pyplot as plt
import pickle

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import dask.array as da
from dask.distributed import Client, LocalCluster
import psutil
import gc
from utils.data_augment import channel_mixup, trial_mixup, channel_reverse, channel_noise, channel_mixure, use_DWTA, augment_with_CR


# 设置 OpenBLAS 的线程数
os.environ["OPENBLAS_NUM_THREADS"] = "4"
# 设置 MKL 的线程数
os.environ["MKL_NUM_THREADS"] = "4"
# 设置 NumPy 本身的线程数(用于某些依赖的库)
os.environ["NUMEXPR_NUM_THREADS"] = "4"


import numpy as np
import scipy.signal
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from einops import rearrange
def SubBandSplit(data: np.ndarray, freq_start: int = 4, freq_end: int = 40, bandwidth: int = 4, fs: int = 250):
    """
    优化后的子带切分函数
    data(batch,channel,time) --> sub_band_data(batch,(channel*nBands),time)
    """
    @lru_cache(maxsize=32)
    def get_sos_coeffs(freq_low, freq_high, fs):
        """缓存并返回 SOS 滤波器系数"""
        return scipy.signal.butter(6, [2.0 * freq_low / fs, 2.0 * freq_high / fs], 'bandpass', output='sos')

    def process_single_band(args):
        """处理单个频带的数据"""
        data, freq_low, freq_high = args
        sos = get_sos_coeffs(freq_low, freq_high, fs)
        return scipy.signal.sosfilt(sos, data, axis=-1)

    subbands = np.arange(freq_start, freq_end + 1, bandwidth)
    with ThreadPoolExecutor() as executor:
        # 准备每个频带的参数
        band_args = [(data, low_freq, high_freq) 
                     for low_freq, high_freq in zip(subbands[:-1], subbands[1:])]
        # 并行处理每个频带
        results = list(executor.map(process_single_band, band_args))

    # 重塑结果以匹配所需的输出形状
    sub_band_data = np.stack(results, axis=1).astype(np.float32)
    del results
    gc.collect()
    # return rearrange(sub_band_data, 'b c t n -> b (c n) t')
    return sub_band_data


def split(x, y):
    idx = np.arange(len(x))
    train_size = 240
    return x[idx[:train_size]], y[idx[:train_size]], x[idx[train_size:]], y[idx[train_size:]]

# 保存4s数据,任务标签和用户标签
def IDLoader14004(dataroot="/mnt/data1/tyl/data/BNCI2014004/data/",split:str="T", MI_time:int=4):
    # 9 subjects in 14004
    ids = list(range(1,10,1))
    m, n, p = None, None, None
    for id in ids:
        # 对原始数据进行处理，训练集数据
        raweeg = scio.loadmat(dataroot + f"B0{id}{split}.mat")["data"] # shape(1,3) three sessions:120 + 120 + 160 trials
        sessions_num = raweeg.shape[1]
        fs = raweeg[0][0]["fs"][0][0][0][0] # 250hz
        t_data, s_label = [], []
        for session_idx in range(sessions_num): # 取出每个session数据和标签
            slice_data = []
            data = raweeg[0][session_idx]["X"][0][0].T # (channels,samples)
            trigger = raweeg[0][session_idx]["trial"][0][0].T.squeeze(axis=0) # trigger
            trial_num = len(trigger)
            label = raweeg[0][session_idx]["y"][0][0].T.squeeze(axis=0) - 1 # 0为左手，1为右手
            # remove the EOG signal
            data = data[0:3, :]
            s_label.append(label)
            for idx in range(trial_num): 
                # mat index begin from 1 while python from 0
                # trial tigger begin from fixation cross
                position = int(trigger[idx] - 1 + fs * 3)
                slice_data.append(data[:, position : position + fs * MI_time]) 
            if session_idx == 0:
                t_data.append(slice_data)
                t_data = np.squeeze(t_data)          
            else:
                t_data = np.concatenate((t_data , np.array(slice_data)))
        s_label = np.concatenate(s_label)
        t_label = np.full(len(t_data),fill_value=int(id))
        if (m is None) and (n is None):
            m,n,p = np.array(t_data), np.array(t_label), s_label
        else:
            m,n,p = np.concatenate((m,t_data)), np.concatenate((n,t_label)), np.concatenate((p,s_label))  
        # slightly class imbalance:
        # (400,400,400,420,420,400,400,440,400) for train, 3680 in total
        # (320,280,320,320,320,320,320,320,320) for test, 2840 in total
    # 9个用户的数据(m)、用户id标签(n)和任务标签(p)
    return m, n, p

# 保存4s数据,任务标签和用户标签
def IDLoader14001(dataroot="/mnt/data1/tyl/data/BNCI2014001/data/",split:str="T", MI_time:int=4):
    # 9 subjects in 14001
    ids = list(range(1,10,1))
    m, n, p = None, None, None
    for id in ids:
        if id == 4 and split == "T":
            file_list = [1,2,3,4,5,6] # A04T只有7个cell
        else :
            file_list = [3,4,5,6,7,8] # 后6个session才有数据
        # 对原始数据进行处理，训练集数据
        raweeg = scio.loadmat(os.path.join(dataroot,f"A0{id}{split}.mat"))["data"] # 48*6=288
        fs = raweeg[0][0]["fs"][0][0][0][0] # 250Hz
        t_data, s_label = [], []
        for session_idx in file_list:
            data = raweeg[0][session_idx]["X"][0][0].T # (channels,points)
            trigger = raweeg[0][session_idx]["trial"][0][0].squeeze()
            label = raweeg[0][session_idx]["y"][0][0].squeeze() - 1 
            trial_num = len(trigger)
            # channel selection:the last three channels were EOG siganls
            data = np.array(data[:-3, :])
            s_label.append(label)
            one_session_data = np.zeros((trial_num,data.shape[0],fs * MI_time))
            # get one session data
            for idx in range(trial_num):
                position = int(trigger[idx] -1 + fs * 2)
                one_session_data[idx,:,:] = data[:, position : position + MI_time * fs]
            t_data.append(one_session_data)
            # all train data:(288,22,750)
            if session_idx ==8: 
                t_data = np.array(t_data).reshape((-1,data.shape[0],fs * MI_time))
            if file_list[0] == 1 and session_idx == 6:
                t_data = np.array(t_data).reshape((-1,data.shape[0],fs * MI_time))
        t_label = np.full(len(t_data),fill_value=int(id))
        s_label = np.concatenate(s_label)
        if (m is None) and (n is None):
            m,n,p = np.array(t_data), np.array(t_label), s_label
        else:
            m,n,p = np.concatenate((m,t_data)), np.concatenate((n,t_label)), np.concatenate((p,s_label))  
    return m, n, p

# 获取BCI Database 静息态数据
def GetloaderRest85(seed):
    raw = np.load("/mnt/data1/tyl/UserID/dataset/mydata/rest_BCI85.pkl", allow_pickle=True, mmap_mode='r')
    data1, data2 = raw["ori_train_x"], raw["ori_test_x"]
    label1, label2 = raw["ori_train_s"], raw["ori_test_s"]
    data1 , data2 = data1 * 1e5 , data2 * 1e5
    data1, data2 = [data.reshape((-1,data.shape[-2],data.shape[-1])) for data in [data1, data2]]
    label1, label2 = [label.reshape((-1)) for label in [label1, label2]]
    DataProcessor = preprocessing(fs=512) 
    [data1, data2] = [DataProcessor.EEGpipline(x) for x in [data1, data2]] 
    # 将数据降采样到250Hz
    [data1, data2] = [mne.filter.resample(x, 250, 512, npad='auto') for x in [data1, data2]] 
    tx, vx, ts, vs = train_test_split(data1, label1, test_size=0.2, random_state=seed, stratify=label1)
    [tx, vx, data2] = [np.expand_dims(x,axis=1) for x in [tx, vx, data2]] # if EEGNet, DeepConvNet or ShallowConvNet
    print(f"数据比例-----训练集:验证集:测试集 = {tx.shape}:{vx.shape}:{data2.shape}")
    [trainloader, validateloader, testloader] = [trans_torch(x,y,mode) \
        for x,y,mode in zip([tx, vx, data2], [ts, vs, label2], ["train","test","test"])]
    return trainloader, validateloader, testloader



# 跨任务，静息/任务分别作训练测试
def Crossloader85(seed):
    rawbci = np.load("/mnt/data1/tyl/UserID/dataset/mydata/ori_BCI85.pkl",allow_pickle=True)
    rawrest = np.load("/mnt/data1/tyl/UserID/dataset/mydata/rest_BCI85.pkl",allow_pickle=True)
    # # cross bci_rest
    # data1, label1 = rawbci["ori_train_x"], rawbci["ori_train_s"]
    # data2, label2 = rawrest["ori_train_x"], rawrest["ori_train_s"]
    # data2 *= 1e5
    # cross rest_bci
    data1, label1 = rawrest["ori_train_x"], rawrest["ori_train_s"]
    data2, label2 = rawbci["ori_train_x"], rawbci["ori_train_s"]
    data1 *= 1e5
    data1, data2 = [data.reshape((-1,data.shape[-2],data.shape[-1])) for data in [data1, data2]]
    label1, label2 = [label.reshape((-1)) for label in [label1, label2]]
    DataProcessor = preprocessing(fs=512) 
    [data1, data2] = [DataProcessor.EEGpipline(x) for x in [data1, data2]] 
    tx, vx, ts, vs = train_test_split(data1, label1, test_size=0.2, random_state=seed, stratify=label1)
    [tx, vx, data2] = [np.expand_dims(x,axis=1) for x in [tx, vx, data2]] # if EEGNet, DeepConvNet or ShallowConvNet
    [trainloader, validateloader, testloader] = [trans_torch(x,y) for x,y in zip([tx, vx, data2], [ts, vs, label2])]
    return trainloader, validateloader, testloader



def GetLoader14xxx(seed, split:str="001"):
    data = scio.loadmat(f"/mnt/data1/tyl/UserID/dataset/mydata/ori_{split}.mat") # split = "001", "004"
    # 训练集, 测试集
    data1,label1 = data["ori_train_x"], data["ori_train_s"]
    data2,label2 = data["ori_test_x"], data["ori_test_s"]  
    [label1 ,label2] = [label.squeeze() for label in [label1 ,label2]]
    DataProcessor = preprocessing(fs=250) # 001/004采样率
    [data1, data2] = [DataProcessor.EEGpipline(x) for x in [data1, data2]] 
    tx, vx, ty, vy = train_test_split(data1, label1, test_size=0.2, random_state=seed, stratify=label1) 
    # [tx, vx, data2] = [SubBandSplit(x,8,32,2) for x in [tx, vx, data2]]
    [tx, vx, data2] = [np.expand_dims(x,axis=1) for x in [tx, vx, data2]] 
    print(f"数据比例-----训练集:验证集:测试集 = {tx.shape}:{vx.shape}:{data2.shape}")
    [trainloader, validateloader, testloader] = [trans_torch(x,y,mode) \
        for x,y,mode in zip([tx, vx, data2], [ty, vy, label2], ["train","test","test"])]
    del data1, data2, label1, label2, tx, vx, ty, vy
    gc.collect()
    return trainloader, validateloader, testloader


def GetloaderBCI85(seed):
    def load_data(file_path):
        return np.load(file_path, allow_pickle=True, mmap_mode='r')
    def resample_data(x):
        return mne.filter.resample(x, 250, 512, npad='auto')
    def process_data(data, processor):
        processed_data = np.empty_like(data)
        chunk_size = min(data.shape[0], max_chunk_size)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(0, data.shape[0], chunk_size):
                chunk = data[i:i+chunk_size]
                future = executor.submit(processor, chunk)
                futures.append((i, future))
            for i, future in futures:
                processed_data[i:i+chunk_size] = future.result()
        return processed_data

    # 获取可用的CPU核心数和内存
    num_cores = psutil.cpu_count(logical=False)
    available_memory = psutil.virtual_memory().available
    max_chunk_size = int((available_memory // (4 * num_cores)) // 16)  # 假设使用32位浮点数(4bytes),1/16减少内存开销占用

    # 加载数据
    data = load_data("/mnt/data1/tyl/UserID/dataset/mydata/ori_BCI85.pkl")
    data1, ex = data["ori_train_x"], data["ori_test_x"]
    data1, ex = [x.reshape((-1, x.shape[-2], x.shape[-1])) for x in [data1, ex]]
    label1, es = data["ori_train_s"], data["ori_test_s"]
    label1, es = [s.reshape(-1) for s in [label1, es]]
    # 数据处理
    DataProcessor = preprocessing(fs=512)
    processor = DataProcessor.EEGpipline
    data1, ex = [process_data(x, processor) for x in [data1, ex]]
    data1, ex = [resample_data(x) for x in [data1, ex]]
    # 数据分割
    tx, vx, ts, vs = train_test_split(data1, label1, test_size=0.2, random_state=seed, stratify=label1)
    tx, vx, ex = [np.expand_dims(x, axis=1) for x in [tx, vx, ex]]
    print(f"数据比例-----训练集:验证集:测试集 = {tx.shape}:{vx.shape}:{ex.shape}")
    [trainloader, validateloader, testloader] = [trans_torch(x, y, mode) for x, y, mode in 
                                                 zip([tx, vx, ex], [ts, vs, es], ["train", "test", "test"])]
    
    return trainloader, validateloader, testloader


def GetloaderLJ30(seed,aug_type=None):
    data = scio.loadmat("/mnt/data1/tyl/UserID/dataset/mydata/ori_LingJiu30.mat")
    data1, ex = data["ori_train_x"], data["ori_test_x"]
    data1, ex = [x.reshape((-1,x.shape[-2],x.shape[-1])) for x in [data1, ex]] # (6300/3150,20,1500)
    label1, es = data["ori_train_s"], data["ori_test_s"]
    label1, es = [s.reshape(-1) for s in [label1, es]] # (6300/3150,)
    DataProcessor = preprocessing(fs=300) 
    [data1, ex] = [DataProcessor.EEGpipline(x) for x in [data1, ex]] 
    tx, vx, ts, vs = train_test_split(data1, label1, test_size=0.2, random_state=seed, stratify=label1)
    # [tx, vx, ex] = [SubBandSplit(x,8,32,2) for x in [tx, vx, ex]]
    #  channel_mixup, trial_mixup, channel_reverse, channel_noise, channel_mixure, use_DWTA, augment_with_CR
    if aug_type is not None:
        x_ori, _,  y_ori, _ = train_test_split(tx, ts, test_size=0.5, random_state=seed, stratify=ts)
        aug_multiplier = 1 # 增强倍数
        if aug_type == "channel_mixup":
            x_aug , y_aug = channel_mixup(data=x_ori, labels=y_ori, multiplier=aug_multiplier)
        elif aug_type == "trial_mixup":
            x_aug , y_aug = trial_mixup(data=x_ori, labels=y_ori, multiplier=aug_multiplier)
        elif aug_type == "channel_reverse":
            x_aug , y_aug = channel_reverse(data=x_ori, labels=y_ori, multiplier=aug_multiplier)
        elif aug_type == "channel_noise":
            x_aug , y_aug = channel_noise(data=x_ori, labels=y_ori, multiplier=aug_multiplier, noise_type="pink")
        elif aug_type == "channel_mixure":
            x_aug , y_aug = channel_mixure(data=x_ori, labels=y_ori, multiplier=aug_multiplier)
        elif aug_type == "use_DWTA":
            x_aug , y_aug = use_DWTA(data=x_ori, labels=y_ori, multiplier=aug_multiplier)
        elif aug_type == "augment_with_CR":
            x_aug , y_aug = augment_with_CR(x=x_ori, y=y_ori)
        else:
            x_aug, y_aug = x_ori, y_ori
        tx, ts = np.concatenate([tx, x_aug], axis=0), np.concatenate([ts, y_aug], axis=0)

    [tx, vx, ex] = [np.expand_dims(x,axis=1) for x in [tx, vx, ex]] # if EEGNet, DeepConvNet or ShallowConvNet
    print(f"数据比例-----训练集:验证集:测试集 = {tx.shape}:{vx.shape}:{ex.shape}")
    [trainloader, validateloader, testloader] = [trans_torch(x,y,mode) for x,y,mode in \
                                                 zip([tx, vx, ex], [ts, vs, es], ["train","test","test"])]
    del data1, ex, label1, es, tx, vx, ts, vs
    gc.collect()
    return trainloader, validateloader, testloader


# MI:(54,200,62,4000) --> downsample: (54,200,62,1000)
# SSVEP:(54,200,62,4000) --> downsample : (54,200,62,1000)
# Rest:(54,180,62,4000) --> downsample : (54,180,62,1000)
# ERP:(54,4140,62,800) --> reduce the trial_num : (54,200,62,800)
def GetLoaderOpenBMI(seed, split:str="MI",aug_type=None): # split = "ERP", "MI", "SSVEP", "Rest"
    def load_data(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    def process_data(data):
        processed_data = np.empty_like(data)
        chunk_size = min(data.shape[0], max_chunk_size)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(0, data.shape[0], chunk_size):
                chunk = data[i:i+chunk_size]
                future = executor.submit(processor, chunk)
                futures.append((i, future))
            for i, future in futures:
                processed_data[i:i+chunk_size] = future.result()
        return processed_data

    # 获取可用的CPU核心数
    num_cores = psutil.cpu_count(logical=False)
    available_memory = psutil.virtual_memory().available
    max_chunk_size = int((available_memory // (4 * num_cores)) // 16)  # 假设使用32位浮点数(4bytes),1/16减少内存开销占用
    # different session for train and test
    data1 = load_data(f'/mnt/data1/tyl/data/OpenBMI/processed/{split}/train.pkl')
    data2 = load_data(f'/mnt/data1/tyl/data/OpenBMI/processed/{split}/test.pkl')
    train_x, train_s = data1['ori_train_x'].astype(np.float32), (data1['ori_train_s']-1).astype(np.int16)
    test_x, test_s = data2['ori_test_x'].astype(np.float32), (data2['ori_test_s']-1).astype(np.int16)

    # ERP: reduce the trial_num, to keep up with the trial_num of other paradigms(MI,SSVEP,Rest)
    fs = 1000 if split == "ERP" else 250
    train_x, test_x = [x.reshape((-1,x.shape[-2],x.shape[-1])) for x in [train_x, test_x]]
    train_s, test_s = [s.reshape(-1) for s in [train_s, test_s]]

    DataProcessor = preprocessing(fs=fs) # MI,SSVEP,Rest:250hz(4s), ERP:1000hz(0.8s)
    processor = DataProcessor.EEGpipline

    train_x, test_x = [process_data(x) for x in [train_x, test_x]]
    tx, vx, ts, vs = train_test_split(train_x, train_s, test_size=0.2, random_state=seed, stratify=train_s)
    #  channel_mixup, trial_mixup, channel_reverse, channel_noise, channel_mixure, use_DWTA, augment_with_CR
    if aug_type is not None:
        x_ori, _,  y_ori, _ = train_test_split(tx, ts, test_size=0.5, random_state=seed, stratify=ts)
        aug_multiplier = 1 # 增强倍数
        if aug_type == "channel_mixup":
            x_aug , y_aug = channel_mixup(data=x_ori, labels=y_ori, multiplier=aug_multiplier)
        elif aug_type == "trial_mixup":
            x_aug , y_aug = trial_mixup(data=x_ori, labels=y_ori, multiplier=aug_multiplier)
        elif aug_type == "channel_reverse":
            x_aug , y_aug = channel_reverse(data=x_ori, labels=y_ori, multiplier=aug_multiplier)
        elif aug_type == "channel_noise":
            x_aug , y_aug = channel_noise(data=x_ori, labels=y_ori, multiplier=aug_multiplier, noise_type="pink")
        elif aug_type == "channel_mixure":
            x_aug , y_aug = channel_mixure(data=x_ori, labels=y_ori, multiplier=aug_multiplier)
        elif aug_type == "use_DWTA":
            x_aug , y_aug = use_DWTA(data=x_ori, labels=y_ori, multiplier=aug_multiplier)
        elif aug_type == "augment_with_CR":
            x_aug , y_aug = augment_with_CR(x=x_ori, y=y_ori)
        else:
            x_aug, y_aug = x_ori, y_ori
        tx, ts = np.concatenate([tx, x_aug], axis=0), np.concatenate([ts, y_aug], axis=0)

    tx, vx, test_x = [np.expand_dims(x, axis=1) for x in [tx, vx, test_x]] # if EEGNet, DeepConvNet or ShallowConvNet
    print(f"数据比例-----训练集:验证集:测试集 = {tx.shape}:{vx.shape}:{test_x.shape}")
    [trainloader, validateloader, testloader] = [trans_torch(x,y,mode) for x,y,mode \
                                                 in zip([tx, vx, test_x], [ts, vs, test_s], ["train","test","test"])]
    return trainloader, validateloader, testloader


"""
Clibration#---------------------------------------------- Test
Rest (600, 65, 1000) (600,) -- 20 * 30 (subs * trials)
Transient (1791, 65, 1000) (1791,) -- 20 * (88~90)
Steady (740, 65, 1000) (740,) -- 20 * 37
P300 (299, 65, 1000) (299,) -- 20 * (15~14)
Motor (2400, 65, 1000) (2400,) -- 20 * 120
SSVEP_SA (240, 65, 1000) (240,) -- 20 * 12

Partial Enrollment#---------------------------------------------- Real Train
Rest (1200, 65, 1000) (1200,) -- 20 * 60 (subs * trials)
Transient (3590, 65, 1000) (3590,) -- 20 * (178~180)
Steady (1530, 65, 1000) (1530,) -- 20 * (75/105)
P300 (599, 65, 1000) (599,) -- 20 * (29~30)
Motor (4828, 65, 1000) (4828,) -- 20* (239~243 / 265)
SSVEP_SA (480, 65, 1000) (480,) -- 20 * 24
! 注意: 去除EasyCap后是64通道
"""
# split = "M3CV_Rest"， "M3CV_Transient", "M3CV_Steady", "M3CV_P300", "M3CV_Motor", "M3CV_SSVEP_SA"
def GetLoaderM3CV(seed, split:str="M3CV_Rest",aug_type=None): 
    def load_data(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    def process_data(data):
        processed_data = np.empty_like(data)
        chunk_size = min(data.shape[0], max_chunk_size)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(0, data.shape[0], chunk_size):
                chunk = data[i:i+chunk_size]
                future = executor.submit(processor, chunk)
                futures.append((i, future))
            for i, future in futures:
                processed_data[i:i+chunk_size] = future.result()
        return processed_data

    # 获取可用的CPU核心数
    num_cores = psutil.cpu_count(logical=False)
    available_memory = psutil.virtual_memory().available
    max_chunk_size = int((available_memory // (4 * num_cores)) // 16)  # 假设使用32位浮点数(4bytes),1/16减少内存开销占用
    split = split[5:]
    # different session for train and test
    data1 = load_data(f'/mnt/data1/tyl/data/M3CV/Train/T_{split}.pkl')
    data2 = load_data(f'/mnt/data1/tyl/data/M3CV/Test/{split}.pkl')
    # 去除EasyCap
    train_x, train_s = data1['data'][:,:-1,:].astype(np.float32), (data1['label']).astype(np.int16)
    test_x, test_s = data2['data'][:,:-1,:].astype(np.float32), (data2['label']).astype(np.int16)

    DataProcessor = preprocessing(fs=250) # MI,SSVEP,Rest:250hz(4s), ERP:1000hz(0.8s)
    processor = DataProcessor.EEGpipline

    train_x, test_x = [process_data(x) for x in [train_x, test_x]]
    tx, vx, ts, vs = train_test_split(train_x, train_s, test_size=0.2, random_state=seed, stratify=train_s)
    #  channel_mixup, trial_mixup, channel_reverse, channel_noise, channel_mixure, use_DWTA, augment_with_CR
    if aug_type is not None:
        x_ori, _,  y_ori, _ = train_test_split(tx, ts, test_size=0.5, random_state=seed, stratify=ts)
        aug_multiplier = 1 # 增强倍数
        if aug_type == "channel_mixup":
            x_aug , y_aug = channel_mixup(data=x_ori, labels=y_ori, multiplier=aug_multiplier)
        elif aug_type == "trial_mixup":
            x_aug , y_aug = trial_mixup(data=x_ori, labels=y_ori, multiplier=aug_multiplier)
        elif aug_type == "channel_reverse":
            x_aug , y_aug = channel_reverse(data=x_ori, labels=y_ori, multiplier=aug_multiplier)
        elif aug_type == "channel_noise":
            x_aug , y_aug = channel_noise(data=x_ori, labels=y_ori, multiplier=aug_multiplier, noise_type="pink")
        elif aug_type == "channel_mixure":
            x_aug , y_aug = channel_mixure(data=x_ori, labels=y_ori, multiplier=aug_multiplier)
        elif aug_type == "use_DWTA":
            x_aug , y_aug = use_DWTA(data=x_ori, labels=y_ori, multiplier=aug_multiplier)
        elif aug_type == "augment_with_CR":
            x_aug , y_aug = augment_with_CR(x=x_ori, y=y_ori)
        else:
            x_aug, y_aug = x_ori, y_ori
        tx, ts = np.concatenate([tx, x_aug], axis=0), np.concatenate([ts, y_aug], axis=0)

    # [tx, vx, test_x] = [SubBandSplit(x,8,32,2) for x in [tx, vx, test_x]]
    tx, vx, test_x = [np.expand_dims(x, axis=1) for x in [tx, vx, test_x]] # if EEGNet, DeepConvNet or ShallowConvNet
    print(f"数据比例-----训练集:验证集:测试集 = {tx.shape}:{vx.shape}:{test_x.shape}")
    [trainloader, validateloader, testloader] = [trans_torch(x,y,mode) for x,y,mode \
                                                 in zip([tx, vx, test_x], [ts, vs, test_s], ["train","test","test"])]
    del data1, data2, train_x, train_s, test_x, test_s, tx, vx, ts, vs
    gc.collect()
    return trainloader, validateloader, testloader




def GetLoaderSEED(seed): 
    def load_data(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    def process_data(data):
        processed_data = np.empty_like(data)
        chunk_size = min(data.shape[0], max_chunk_size)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(0, data.shape[0], chunk_size):
                chunk = data[i:i+chunk_size]
                future = executor.submit(processor, chunk)
                futures.append((i, future))
            for i, future in futures:
                processed_data[i:i+chunk_size] = future.result()
        return processed_data

    # 获取可用的CPU核心数
    num_cores = psutil.cpu_count(logical=False)
    available_memory = psutil.virtual_memory().available
    max_chunk_size = int((available_memory // (4 * num_cores)) // 16)  # 假设使用32位浮点数(4bytes),1/16减少内存开销占用
    # different session for train and test
    data1 = load_data('/mnt/data1/tyl/data/SEED/processed/train.pkl')
    data2 = load_data('/mnt/data1/tyl/data/SEED/processed/test.pkl')
    train_x, train_s = data1['data'].astype(np.float32), (data1['label']).astype(np.int16) -1
    test_x, test_s = data2['data'].astype(np.float32), (data2['label']).astype(np.int16) -1

    DataProcessor = preprocessing(fs=200) 
    processor = DataProcessor.EEGpipline

    train_x, test_x = [process_data(x) for x in [train_x, test_x]]
    # train_x, test_x = [SubBandSplit(x,8,32,2) for x in [train_x, test_x]]
    tx, vx, ts, vs = train_test_split(train_x, train_s, test_size=0.2, random_state=seed, stratify=train_s)
    tx, vx, test_x = [np.expand_dims(x, axis=1) for x in [tx, vx, test_x]] # if EEGNet, DeepConvNet or ShallowConvNet
    print(f"数据比例-----训练集:验证集:测试集 = {tx.shape}:{vx.shape}:{test_x.shape}")
    [trainloader, validateloader, testloader] = [trans_torch(x,y,mode) for x,y,mode \
                                                 in zip([tx, vx, test_x], [ts, vs, test_s], ["train","test","test"])]
    return trainloader, validateloader, testloader




""" 
A, B, C = GetLoader14xxx(2024,"001")
数据比例-----训练集:验证集:测试集 = (2073, 1, 22, 1000):(519, 1, 22, 1000):(2592, 1, 22, 1000)
--- 10.349206447601318 seconds ---
A, B, C = GetLoader14xxx(2024,"004")
数据比例-----训练集:验证集:测试集 = (2944, 1, 3, 1000):(736, 1, 3, 1000):(2840, 1, 3, 1000)
--- 12.539725303649902 seconds ---
A, B, C = GetloaderBCI85(2024)
数据比例-----训练集:验证集:测试集 = (10880, 1, 27, 1250):(2720, 1, 27, 1250):(6800, 1, 27, 1250)
--- 306.2780680656433 seconds ---
A, B, C = GetloaderRest85(2024)
数据比例-----训练集:验证集:测试集 = (2448, 1, 27, 1250):(612, 1, 27, 1250):(3060, 1, 27, 1250)
--- 390.04875445365906 seconds ---
A, B, C = GetloaderLJ30(2024)
数据比例-----训练集:验证集:测试集 = (5040, 1, 20, 1500):(1260, 1, 20, 1500):(3150, 1, 20, 1500)
--- 418.7900390625 seconds ---
A, B, C = GetLoaderOpenBMI(2024,"MI")
数据比例-----训练集:验证集:测试集 = (8640, 1, 62, 1000):(2160, 1, 62, 1000):(10800, 1, 62, 1000)
--- 596.3773910999298 seconds ---
# A, B, C = GetLoaderOpenBMI(2024,"ERP")
A, B, C = GetLoaderOpenBMI(2024,"SSVEP")
数据比例-----训练集:验证集:测试集 = (8640, 1, 62, 1000):(2160, 1, 62, 1000):(10800, 1, 62, 1000)
--- 721.5216789245605 seconds ---
A, B, C = GetLoaderOpenBMI(2024,"Rest")
数据比例-----训练集:验证集:测试集 = (7776, 1, 62, 1000):(1944, 1, 62, 1000):(9720, 1, 62, 1000)
--- 876.2816712856293 seconds ---
A, B, C = GetLoaderOpenBMI_crossparadigm(2024)
数据比例-----训练集:验证集:测试集 = (8640, 1, 62, 1000):(2160, 1, 62, 1000):(10800, 1, 62, 1000)
--- 981.9834458827972 seconds ---
"""


# split = "M3CV_Rest"， "M3CV_Transient", "M3CV_Steady", "M3CV_P300", "M3CV_Motor", "M3CV_SSVEP_SA"
# import time
# t1 = time.time()
# A, B, C = GetLoaderOpenBMI(2024,split="ERP")
# t2 = time.time()
# print(f"--- {t2-t1:.4f} seconds ---")

def M3CV_crosstask(seed, split:list[str]=["M3CV_Rest", "M3CV_Transient"], cross_session:bool=True, session_num=None): 
    def load_data(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    assert isinstance(split, list) and len(split) == 2, "split must be a (str)list with 2 elements"
    task1, task2 = split[0][5:], split[1][5:]
    # different tasks merge into one session
    [data11, data12] = [load_data(f'/mnt/data1/tyl/data/M3CV/{mode}{task1}.pkl') for mode in ["Train/T_", "Test/"]]
    [data21, data22] = [load_data(f'/mnt/data1/tyl/data/M3CV/{mode}{task2}.pkl') for mode in ["Train/T_", "Test/"]]
    task1_x1, task1_s1 = data11['data'][:,:-1,:].astype(np.float32), (data11['label']).astype(np.int16)
    task1_x2, task1_s2 = data12['data'][:,:-1,:].astype(np.float32), (data12['label']).astype(np.int16)
    task2_x1, task2_s1 = data21['data'][:,:-1,:].astype(np.float32), (data21['label']).astype(np.int16)
    task2_x2, task2_s2 = data22['data'][:,:-1,:].astype(np.float32), (data22['label']).astype(np.int16)
    if cross_session:
        train_x, train_s = np.concatenate([task1_x1, task1_x2], axis=0), np.concatenate([task1_s1, task1_s2], axis=0)
        test_x, test_s = np.concatenate([task2_x1, task2_x2], axis=0), np.concatenate([task2_s1, task2_s2], axis=0)
        del data11, data12, data21, data22, task1_x1, task1_x2, task1_s1, task1_s2, task2_x1, task2_x2, task2_s1, task2_s2
    elif not cross_session and session_num is not None:
        if session_num == 1:
            train_x, train_s = task1_x1, task1_s1
            test_x, test_s = task2_x1, task2_s1
        elif session_num == 2:
            train_x, train_s = task1_x2, task1_s2
            test_x, test_s = task2_x2, task2_s2
        elif session_num == 12:
            train_x, train_s = task1_x1, task1_s1
            test_x, test_s = task2_x2, task2_s2
        elif session_num == 21:
            train_x, train_s = task1_x2, task1_s2
            test_x, test_s = task2_x1, task2_s1

    DataProcessor = preprocessing(fs=250) # MI,SSVEP,Rest:250hz(4s), ERP:1000hz(0.8s)
    processor = DataProcessor.EEGpipline

    train_x, test_x = [processor(x) for x in [train_x, test_x]]
    tx, vx, ts, vs = train_test_split(train_x, train_s, test_size=0.2, random_state=seed, stratify=train_s)
    # [tx, vx, test_x] = [SubBandSplit(x,8,32,2) for x in [tx, vx, test_x]]
    tx, vx, test_x = [np.expand_dims(x, axis=1) for x in [tx, vx, test_x]] # if EEGNet, DeepConvNet or ShallowConvNet
    print(f"数据比例-----训练集:验证集:测试集 = {tx.shape}:{vx.shape}:{test_x.shape}")
    [trainloader, validateloader, testloader] = [trans_torch(x,y,mode) for x,y,mode \
                                                 in zip([tx, vx, test_x], [ts, vs, test_s], ["train","test","test"])]
    del train_x, train_s, test_x, test_s, tx, vx, ts, vs, DataProcessor, processor
    gc.collect()
    return trainloader, validateloader, testloader



def OpenBMI_crosstask(seed, split:list[str]=["MI", "ERP"], cross_session:bool=True, session_num=None): 
    def load_data(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    assert isinstance(split, list) and len(split) == 2, "split must be a (str)list with 2 elements"
    task1, task2 = split[0], split[1]
    # different tasks merge into one session
    [data11, data12] = [load_data(f'/mnt/data1/tyl/data/OpenBMI/processed/{task1}/{mode}.pkl') for mode in ["train", "test"]]
    [data21, data22] = [load_data(f'/mnt/data1/tyl/data/OpenBMI/processed/{task2}/{mode}.pkl') for mode in ["train", "test"]]
    task1_x1, task1_s1 = data11['ori_train_x'].astype(np.float32), (data11['ori_train_s']).astype(np.int16) -1
    task1_x2, task1_s2 = data12['ori_test_x'].astype(np.float32), (data12['ori_test_s']).astype(np.int16) -1
    task2_x1, task2_s1 = data21['ori_train_x'].astype(np.float32), (data21['ori_train_s']).astype(np.int16) -1
    task2_x2, task2_s2 = data22['ori_test_x'].astype(np.float32), (data22['ori_test_s']).astype(np.int16) -1
    task1_x1, task1_x2, task2_x1, task2_x2 = [x.reshape((-1,x.shape[-2],x.shape[-1])) for x in [task1_x1, task1_x2, task2_x1, task2_x2]]
    task1_s1, task1_s2, task2_s1, task2_s2 = [s.reshape((-1,)) for s in [task1_s1, task1_s2, task2_s1, task2_s2]]

    if "ERP" in [task1, task2]:
        if task1 == "ERP":
            task2_x1 ,task2_x2 = task2_x1[:,:,100:-100], task2_x2[:,:,100:-100]
        elif task2 == "ERP":
            task1_x1 ,task1_x2 = task1_x1[:,:,100:-100], task1_x2[:,:,100:-100]
    if cross_session:
        train_x, train_s = np.concatenate([task1_x1, task1_x2], axis=0), np.concatenate([task1_s1, task1_s2], axis=0)
        test_x, test_s = np.concatenate([task2_x1, task2_x2], axis=0), np.concatenate([task2_s1, task2_s2], axis=0)
        del data11, data12, data21, data22, task1_x1, task1_x2, task1_s1, task1_s2, task2_x1, task2_x2, task2_s1, task2_s2
    elif not cross_session and session_num is not None:
        if session_num == 1:
            train_x, train_s = task1_x1, task1_s1
            test_x, test_s = task2_x1, task2_s1
        elif session_num == 2:
            train_x, train_s = task1_x2, task1_s2
            test_x, test_s = task2_x2, task2_s2
        elif session_num == 12:
            train_x, train_s = task1_x1, task1_s1
            test_x, test_s = task2_x2, task2_s2
        elif session_num == 21:
            train_x, train_s = task1_x2, task1_s2
            test_x, test_s = task2_x1, task2_s1

    DataProcessor = preprocessing(fs=250) # MI,SSVEP,Rest:250hz(4s), ERP:1000hz(0.8s)
    processor = DataProcessor.EEGpipline
    
    # 根据范式滤波预处理频率不一样
    if "ERP" in [task1, task2]:
        ERPprocessor = preprocessing(fs=1000).EEGpipline
        if task1 == "ERP":
            train_x = ERPprocessor(train_x)
            test_x = processor(test_x)
        if task2 == "ERP":
            train_x = processor(train_x)
            test_x = ERPprocessor(test_x)
    else:
        train_x, test_x = [processor(x) for x in [train_x, test_x]]
    
    tx, vx, ts, vs = train_test_split(train_x, train_s, test_size=0.2, random_state=seed, stratify=train_s)
    # [tx, vx, test_x] = [SubBandSplit(x,8,32,2) for x in [tx, vx, test_x]]
    tx, vx, test_x = [np.expand_dims(x, axis=1) for x in [tx, vx, test_x]] # if EEGNet, DeepConvNet or ShallowConvNet
    print(f"数据比例-----训练集:验证集:测试集 = {tx.shape}:{vx.shape}:{test_x.shape}")
    [trainloader, validateloader, testloader] = [trans_torch(x,y,mode) for x,y,mode \
                                                 in zip([tx, vx, test_x], [ts, vs, test_s], ["train","test","test"])]
    del train_x, train_s, test_x, test_s, tx, vx, ts, vs, DataProcessor, processor
    gc.collect()
    return trainloader, validateloader, testloader


""" 
M3CV模态划分:
1. Motor(4828-2400)、Transient(3590-1791) 70%
2. Rest(1200-600)、Steady(1530-740) 60%
3. P300(599-299)、SSVEP_SA(480-240) 50% 

组合:
[Motor, Rest] [Motor, Steady] [Motor, P300] [Motor, SSVEP_SA] 
[Transient, Rest] [Transient, Steady] [Transient, P300] [Transient, SSVEP_SA] 
[Rest, P300] [Rest, SSVEP_SA] [Steady, P300] [Steady, SSVEP_SA] 
"""