""" 
instruction:some data augment methods-
    add random noise; data flipping
    data_scale(strength or weaken); frequency shift
Author:hust-marx2
time: 2023/9/11
lastest:all methods are used for some specific channels
"""

import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import math

# func: add random noise
# args: 
#     data:raw data; C_noise:hyperparameter to control the amplitude of noise;
#     channel_num: how many channels to add guass noise
def random_noise(data, C_noise:int=2, channel_num:int = 1):
    data_shape = np.array(data).shape # (trials,channels,points)
    channels = data_shape[-2]
    channels_selected = np.random.choice(channels, size=channel_num, replace=False)

    lam = np.random.uniform(low=-0.5, high=0.5, size=1)
    # lam = np.random.normal(mu,sigma,size) 高斯噪声
    noise = data[:, channels_selected, :].copy()
    # 对噪声进行缩放，通过除以增益因子 lam 和 C_noise 来控制噪声的强度
    noise = lam * noise.std(axis=-1) / C_noise  # 按照原文献生成的噪声
    noise = np.expand_dims(noise, axis=-1)
    
    # 创建一个包含噪声的副本，添加噪声到选定通道的数据中
    noisy_data = data.copy()
    noisy_data[:, channels_selected, :] = noisy_data[:, channels_selected, :] + noise
    
    return noisy_data

# func: data flipped;min->max,max->min
def data_flipping(data, channel_num:int = 1):
    data_shape = np.array(data).shape
    channels = data_shape[-2]
    channels_selected = np.random.choice(channels, size=channel_num, replace=False)

    max_values = np.max(data, axis=-1, keepdims=True)

    flipped_data = data.copy()
    flipped_data[:, channels_selected, :] = max_values[:, channels_selected, :] - data[:, channels_selected, :]

    return flipped_data

# func: strengthen or weaken the raw data
def data_scale(data, multi:float=0.05, channel_num:int = 1, method:str="strengthen"):
    data_shape = np.array(data).shape
    channels = data_shape[-2]
    channels_selected = np.random.choice(channels, size=channel_num, replace=False)
    
    pre_data = data.copy()

    if method == "strengthen":
        pre_data[:, channels_selected, :] = data[:, channels_selected, :] * (1 + multi)
    elif method == "weaken":
        pre_data[:, channels_selected, :] = data[:, channels_selected, :] * (1 - multi)
    else:
        raise NotImplementedError

    return pre_data

# func: frequency shift
def freq_shift_H(data, C_freq:float=0.2, channel_num:int=1):
    # 通道选择的希尔伯特变换
    data_shape = np.array(data).shape
    channels = data_shape[-2]
    channels_selected = np.random.choice(channels, size=channel_num, replace=False)

    # 对信号进行希尔伯特变换
    analytic_signal = hilbert(data)
    # 计算瞬时相位
    phase = np.unwrap(np.angle(analytic_signal))
    # 计算瞬时频率,采用差分近似
    inst_freq = np.diff(phase) / (2.0 * np.pi)
    # 对瞬时频率进行偏移
    inst_freq_shifted = inst_freq + C_freq
    # 通过瞬时频率和相位重构信号
    phase = np.cumsum(2.0 * np.pi * inst_freq_shifted) / data_shape[-1]
    new_phase = np.zeros(np.prod(data_shape))
    new_phase[-len(phase)-1:-1] = phase[:]
    new_phase = new_phase.reshape(*(data_shape))

    signal_shifted = np.real(np.exp(1j * new_phase) * analytic_signal)
    # signal_shifted = np.abs(analytic_signal) * np.exp(1j * phase)
    return signal_shifted

# 数据的镜像翻转,将左右手的数据进行翻转增强
# author:cxr
def mirror_reverse(DATA,ch_name):
    """
    data:(N,C,T),label(n,):0,1,2
    if label==0: represent left hand: reversed_label==1 right hand
    if label==1: represent right hand: reversed_label==0 left hand
    if label==2: represent feet: reversed_label==2 feet
    reversed_data (N,C,T),reversed_label(n,)
    """
    #生成匹配镜像通道对
    key_reverse=[]
    for name_value in ["Fp","AF","F","FC","FT","C","T","CP","TP","P","PO","O"]:
        for num_valve in [1,3,5,7]:
            key_reverse.append([name_value+ str(num_valve),name_value+ str(num_valve+1)])
    id_reverse=[]
    for pair in key_reverse:
        if pair[0] in ch_name and pair[1] in ch_name:
           id_reverse.append([ch_name.index(pair[0]),ch_name.index(pair[1])])

    #镜像数据
    reversed_data = np.array(DATA).copy()
    for xuhao,data in enumerate(DATA): # 例如(59,1000)
        for id_pair in id_reverse:
            reversed_data[xuhao,id_pair[1],:],reversed_data[xuhao,id_pair[0],:] = data[id_pair[0]],data[id_pair[1]]

    return reversed_data

# # 原始脑电数据信号
# raw_data = np.random.rand(320, 6, 1000)
# # 处理后的信号
# augmented_data = np.random.rand(320, 6, 1000)

# # 绘制原始脑电数据信号和处理后的信号
# fig, axs = plt.subplots(2, 1, figsize=(10, 6))

# # 绘制原始脑电数据信号
# for i in range(6):
#     axs[0].plot(raw_data[:, i, 0], label=f'Channel {i+1}')
# axs[0].set_title('Raw EEG Data')
# axs[0].legend()

# # 绘制处理后的信号
# for i in range(6):
#     axs[1].plot(augmented_data[:, i, 0], label=f'Channel {i+1}')
# axs[1].set_title('Augmented EEG Data')
# axs[1].legend()

# plt.show()

# -*- coding: utf-8 -*-
# @Time    : 2021/6/25 21:01
# @Author  : wenzhang
# @File    : data_augment.py

import numpy as np
from scipy.signal import hilbert


def data_aug(data, labels, size, flag_aug):
    # augments data based on boolean inputs reuse_data, noise_data, neg_data, freq_mod data.
    # data: samples * size * n_channels
    # size: int(freq * window_size)
    # Returns: entire training dataset after data augmentation, and the corresponding labels

    # noise_flag, neg_flag, mult_flag, freq_mod_flag test 75.154
    # mult_flag, noise_flag, neg_flag, freq_mod_flag test 76.235
    # noise_flag, neg_flag, freq_mod_flag test 76.157

    mult_flag, noise_flag, neg_flag, freq_mod_flag = flag_aug[0], flag_aug[1], flag_aug[2], flag_aug[3]

    n_channels = data.shape[2]
    data_out = data  # 1 raw features
    labels_out = labels

    if mult_flag:  # 2 features
        mult_data_add, labels_mult = data_mult_f(data, labels, size, n_channels=n_channels)
        data_out = np.concatenate([data_out, mult_data_add], axis=0)
        labels_out = np.append(labels_out, labels_mult)
    if noise_flag:  # 1 features
        noise_data_add, labels_noise = data_noise_f(data, labels, size, n_channels=n_channels)
        data_out = np.concatenate([data_out, noise_data_add], axis=0)
        labels_out = np.append(labels_out, labels_noise)
    if neg_flag:  # 1 features
        neg_data_add, labels_neg = data_neg_f(data, labels, size, n_channels=n_channels)
        data_out = np.concatenate([data_out, neg_data_add], axis=0)
        labels_out = np.append(labels_out, labels_neg)
    if freq_mod_flag:  # 2 features
        freq_data_add, labels_freq = freq_mod_f(data, labels, size, n_channels=n_channels)
        data_out = np.concatenate([data_out, freq_data_add], axis=0)
        labels_out = np.append(labels_out, labels_freq)

    # 最终输出data格式为
    # raw 144, mult_add 144, mult_reduce 144, noise 144, neg 144, freq1 144, freq2 144
    return data_out, labels_out


def data_noise_f(data, labels, size, n_channels=22):
    new_data = []
    new_labels = []
    noise_mod_val = 2
    # print("noise mod: {}".format(noise_mod_val))
    for i in range(len(labels)):
        if labels[i] >= 0:
            stddev_t = np.std(data[i])
            rand_t = np.random.rand(data[i].shape[0], data[i].shape[1])
            rand_t = rand_t - 0.5
            to_add_t = rand_t * stddev_t / noise_mod_val
            data_t = data[i] + to_add_t
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)

    return new_data_ar, new_labels


def data_mult_f(data, labels, size, n_channels=22):
    new_data = []
    new_labels = []
    mult_mod = 0.05
    # print("mult mod: {}".format(mult_mod))
    for i in range(len(labels)):
        if labels[i] >= 0:
            # print(data[i])
            data_t = data[i] * (1 + mult_mod)
            new_data.append(data_t)
            new_labels.append(labels[i])

    for i in range(len(labels)):
        if labels[i] >= 0:
            data_t = data[i] * (1 - mult_mod)
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)

    return new_data_ar, new_labels


def data_neg_f(data, labels, size, n_channels=22):
    # Returns: data double the size of the input over time, with new data
    # being a reflection along the amplitude

    new_data = []
    new_labels = []
    for i in range(len(labels)):
        if labels[i] >= 0:
            data_t = -1 * data[i]
            data_t = data_t - np.min(data_t)
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)

    return new_data_ar, new_labels


def freq_mod_f(data, labels, size, n_channels=22):
    new_data = []
    new_labels = []
    # print(data.shape)
    freq_mod = 0.2
    # print("freq mod: {}".format(freq_mod))
    for i in range(len(labels)):
        if labels[i] >= 0:
            low_shift = freq_shift(data[i], -freq_mod, num_channels=n_channels)
            new_data.append(low_shift)
            new_labels.append(labels[i])

    for i in range(len(labels)):
        if labels[i] >= 0:
            high_shift = freq_shift(data[i], freq_mod, num_channels=n_channels)
            new_data.append(high_shift)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)

    return new_data_ar, new_labels


def freq_shift(x, f_shift, dt=1 / 250, num_channels=22):
    shifted_sig = np.zeros((x.shape))
    len_x = len(x)
    padding_len = 2 ** nextpow2(len_x)
    padding = np.zeros((padding_len - len_x, num_channels))
    with_padding = np.vstack((x, padding))
    hilb_T = hilbert(with_padding, axis=0)
    t = np.arange(0, padding_len)
    shift_func = np.exp(2j * np.pi * f_shift * dt * t)
    for i in range(num_channels):
        shifted_sig[:, i] = (hilb_T[:, i] * shift_func)[:len_x].real

    return shifted_sig


def nextpow2(x):
    return int(np.ceil(np.log2(np.abs(x))))



import numpy as np
import math

# func: add random noise
# args: 
#     data:raw data; C_noise:hyperparameter to control the amplitude of noise;
#     channel_num: how many channels to add guass noise
def random_noise(data, C_noise:int=2, channel_num:int = 2):
    data_shape = np.array(data).shape # (trials,channels,points)
    channels = data_shape[-2]
    channels_selected = np.random.choice(channels, size=channel_num, replace=False)

    lam = np.random.uniform(low=-0.5, high=0.5, size=1)
    # lam = np.random.normal(mu,sigma,size) 高斯噪声
    noise = data[:, channels_selected, :].copy()
    # 对噪声进行缩放，通过除以增益因子 lam 和 C_noise 来控制噪声的强度
    noise = lam * noise.std(axis=-1) / C_noise  # 按照原文献生成的噪声
    noise = np.expand_dims(noise, axis=-1)
    
    # 创建一个包含噪声的副本，添加噪声到选定通道的数据中
    noisy_data = data.copy()
    noisy_data[:, channels_selected, :] = noisy_data[:, channels_selected, :] + noise
    
    return noisy_data

# func: data flipped;min->max,max->min
def data_flipping(data, channel_num:int = 2):
    data_shape = np.array(data).shape
    channels = data_shape[-2]
    channels_selected = np.random.choice(channels, size=channel_num, replace=False)

    max_values = np.max(data, axis=-1, keepdims=True)

    flipped_data = data.copy()
    flipped_data[:, channels_selected, :] = max_values[:, channels_selected, :] - data[:, channels_selected, :]

    return flipped_data

# func: strengthen or weaken the raw data
def data_scale(data, multi:float=0.05, channel_num:int = 1, method:str="strengthen"):
    data_shape = np.array(data).shape
    channels = data_shape[-2]
    channels_selected = np.random.choice(channels, size=channel_num, replace=False)
    
    pre_data = data.copy()

    if method == "strengthen":
        pre_data[:, channels_selected, :] = data[:, channels_selected, :] * (1 + multi)
    elif method == "weaken":
        pre_data[:, channels_selected, :] = data[:, channels_selected, :] * (1 - multi)

    return pre_data

# 优化版本,提升算法实际运行效率
# (200, 5, 750)滑窗为350，步长为50
# 优化前时长:2.9198s; 优化后时长:0.0173s; 快了168倍
def shard(data, label, postdata_len, strip, if_aug=True, rf=None):
    data,label = np.array(data),np.array(label)
    rf = np.array(rf) if rf is not None else None
    trial_num, chan_num, data_len = data.shape
    pdl = int(postdata_len) if if_aug else data_len
    multis = (data_len - pdl) // strip + 1 if if_aug else 1
    step = strip if if_aug else pdl
    
    # 预分配内存
    seg_data = np.zeros((trial_num * multis, chan_num, pdl))
    seg_label = np.zeros((trial_num * multis,) + label.shape[1:], dtype=label.dtype)
    
    if rf is not None:
        seg_rf = np.zeros((trial_num * multis,) + rf.shape[1:], dtype=rf.dtype)
    
    # 使用numpy的高级索引和切片操作
    for i in range(multis):
        start = i * step
        end = start + pdl
        seg_data[i::multis] = data[:, :, start:end]
        seg_label[i::multis] = label
        if rf is not None:
            seg_rf[i::multis] = rf
    
    if rf is not None:
        return seg_data, seg_label, seg_rf
    else:
        return seg_data, seg_label


# !遗弃版本
# 数据滑窗增强，返回切分好的数据，增加trials的数量
def shard_(data, label, postdata_len, strip, if_aug : bool = True, rf:int=None):
    # args:postdata_len--处理后数据长度--点数，不同数据长度的模型; strip--切分数据段间隔 ;
    # data.shape = (trails,channels,points) 
    pdl = int(postdata_len)
    data = np.array(data) # 可以在这一步进行选择通道
    trial_num, chan_num, data_len = data.shape # 样本数，通道数，每个通道样本点数   
    multis = int(math.floor((data_len-pdl)/strip)) + 1 # 一个trail数据增强后变成几个trails

    if if_aug == True:
        step = strip # 增强就切分
    else:
        pdl = data_len # 不增强直接以原始数据长度切完
    
    # 开始切分,对于每个trail依次切分
    for num in range(trial_num):
        one_trial_seg = np.zeros((multis,chan_num,pdl))
        tmp = data[num,:,:]# 一个试次数据(channels,points)
        tmpy = label[num]
        tmps = rf[num] if rf is not None else None
        i = pdl
        j = 0
        while 1:
            if i > data_len or j > multis-1:
                break
            else :
                one_trial_seg[j,:,:] = tmp[:,i-pdl:i]
                i += step
                j += 1 
        if num==0:
            seg_data = one_trial_seg
            seg_label = np.array([tmpy]*multis) 
            if tmps is not None:
               seg_rf = np.array([tmps]*multis)
        else:
            seg_data = np.concatenate((seg_data,one_trial_seg),axis=0)
            seg_label = np.concatenate((seg_label,np.array([tmpy]*multis)),axis=0)
            if tmps is not None:
               seg_rf = np.concatenate((seg_rf,np.array([tmps]*multis)),axis=0)
    
    if rf is not None:
        return seg_data, seg_label, seg_rf
    else:
        return seg_data, seg_label  


# !遗弃版本
def channel_mixure_(data, channel_num:int = 3, multiplier:int = 4):
    N, C, _ =  data.shape
    aug_data= []
    while len(aug_data) < N * multiplier:
        select_ch = np.random.choice(C, size=channel_num, replace=False)
        select_idx = np.random.choice(N, size=channel_num, replace=False)
        select_trial = np.random.randint(N, size=1)
        new_trial = data[select_trial,:,:].copy().squeeze()
        for ch in range(channel_num):
            new_trial[select_ch[ch],:] = data[select_idx[ch], select_ch[ch], :]
        aug_data.append(new_trial)
    # return np.concatenate((aug_data, data), axis=0)
    return np.array(aug_data)


def channel_mixure(data, labels, num_classes=23, channel_num:int = 5, multiplier:int = 4):
    aug_data, aug_labels = [], []
    channel_num = min(channel_num, int(data.shape[1]//4))

    # for label in range(num_classes):
    for label in np.unique(labels):
        class_data = data[labels == label]
        N, C, T = class_data.shape

        for _ in range(int(N * multiplier)):
            select_ch = np.random.choice(C, size=channel_num, replace=False)
            select_idx = np.random.choice(N, size=channel_num, replace=False)
            select_trial = np.random.randint(N, size=1)
            new_trial = class_data[select_trial,:,:].copy().squeeze()
            for ch in range(channel_num):
                new_trial[select_ch[ch],:] = class_data[select_idx[ch], select_ch[ch], :]
            aug_data.append(new_trial)
            aug_labels.append(label)

    return np.array(aug_data), np.array(aug_labels)


def channel_mixup(data, labels, num_classes=23, multiplier=1):
    aug_data = []
    aug_labels = []

    # for label in range(num_classes):
    for label in np.unique(labels):
        # 获取当前类的数据
        class_data = data[labels == label]
        N, C, T = class_data.shape

        for _ in range(int(N * multiplier)):
            # 随机选择两个样本
            indices = np.random.choice(N, size=2, replace=False)
            sample1 = class_data[indices[0]]
            sample2 = class_data[indices[1]]

            # 随机生成一个0.3到0.7之间的系数
            p = np.random.uniform(0.3, 0.7)
            # 生成新的样本
            new_sample = p * sample1 + (1 - p) * sample2
            # 保存增强后的样本和标签
            aug_data.append(new_sample)
            aug_labels.append(label)

    aug_data = np.array(aug_data)
    aug_labels = np.array(aug_labels)

    # # 合并原始数据和增强数据
    # augmented_data = np.concatenate((data, aug_data), axis=0)
    # augmented_labels = np.concatenate((labels, aug_labels), axis=0)
    # return augmented_data, augmented_labels
    return aug_data, aug_labels

def trial_mixup(data, labels, num_classes=23, multiplier=1):
    aug_data = []
    aug_labels = []

    # for label in range(num_classes):
    for label in np.unique(labels):
        # 获取当前类的数据
        class_data = data[labels == label]
        N, C, T = class_data.shape

        for _ in range(int(N * multiplier)):
            # 随机选择两个样本
            indices = np.random.choice(N, size=2, replace=False)
            sample1 = class_data[indices[0]]
            sample2 = class_data[indices[1]]
            # 将两个trial的前后部分交换
            temp = sample1[:,:T//2]
            sample1[:, :T // 2] = sample2[:, :T // 2]
            sample2[:, :T // 2] = temp
            # 保存增强后的样本和标签
            aug_data.append(sample1)
            aug_labels.append(label)
            aug_data.append(sample2)
            aug_labels.append(label)
    aug_data = np.array(aug_data)
    aug_labels = np.array(aug_labels)
    # # 合并原始数据和增强数据
    # augmented_data = np.concatenate((data, aug_data), axis=0)
    # augmented_labels = np.concatenate((labels, aug_labels), axis=0)
    # return augmented_data, augmented_labels
    return aug_data, aug_labels

def channel_reverse(data, labels, multiplier=1):
    N, C, T = data.shape
    aug_data = []

    # 遍历所有样本，并进行时间反转
    for i in range(N):
        reversed_sample = np.flip(data[i], axis=-1)
        aug_data.append(reversed_sample)

    select_idx = np.random.choice(N, size=int(N * multiplier), replace=False)
    aug_data = np.array(aug_data)[select_idx]
    aug_labels = labels[select_idx]

    # # 合并原始数据和增强数据
    # augmented_data = np.concatenate((data, aug_data), axis=0)
    # augmented_labels = np.concatenate((labels, labels.repeat(multiplier)), axis=0)
    # return augmented_data, augmented_labels
    return aug_data, aug_labels

def channel_noise(data, labels, multiplier=1, noise_type='pink'):
    """
    向数据添加噪声 

    参数:
    - data: 输入数据，形状为 (样本数, 通道数, 时间步数)
    - noise_type: 噪声类型，'gaussian', 'salt_and_pepper', 'poisson', 'pink' 

    返回:
    - 含噪声的数据
    """
    def add_gaussian_noise(data, mean=0.0, std=0.1):
        noise = np.random.normal(mean, std, data.shape)
        return data + noise
    def add_salt_and_pepper_noise(data, salt_prob=0.01, pepper_prob=0.01):
        noisy_data = data.copy()
        num_salt = np.ceil(salt_prob * data.size)
        num_pepper = np.ceil(pepper_prob * data.size)

        # Add salt noise
        coords = [np.random.randint(0, i, int(num_salt)) for i in data.shape]
        noisy_data[tuple(coords)] = np.max(data)

        # Add pepper noise
        coords = [np.random.randint(0, i, int(num_pepper)) for i in data.shape]
        noisy_data[tuple(coords)] = np.min(data)

        return noisy_data
    def add_poisson_noise(data):
        noise = np.random.poisson(size=data.shape)
        return data + noise
    def add_pink_noise(data, alpha=1.0):
        # Generate pink noise using Voss-McCartney algorithm
        num_samples = data.shape[-1]
        num_columns = int(np.ceil(np.log2(num_samples)))
        shape = (data.shape[0], data.shape[1], 2 ** num_columns)
        noise = np.zeros(shape)

        # Generate pink noise
        b = np.random.randn(*shape)
        for i in range(1, num_columns):
            noise[:, :, ::2 ** i] += b[:, :, ::2 ** i]
        noise = noise[:, :, :num_samples]

        # Scale noise
        noise *= (np.arange(num_samples) + 1) ** (-alpha / 2.0)
        return data + noise

    if noise_type == 'gaussian':
        add_noise_func = add_gaussian_noise
    elif noise_type == 'salt_and_pepper':
        add_noise_func = add_salt_and_pepper_noise
    elif noise_type == 'poisson':
        add_noise_func = add_poisson_noise
    elif noise_type == 'pink':
        add_noise_func = add_pink_noise
    else:
        raise ValueError(f"未知的噪声类型: {noise_type}")

    N, C, T = data.shape
    aug_data = []

    for _ in range(multiplier):
        noisy_data = add_noise_func(data)
        aug_data.append(noisy_data)
    aug_data = np.array(aug_data).reshape(multiplier * N, C, T)
    aug_labels = labels.repeat(multiplier)

    # # 合并原始数据和增强数据
    # augmented_data = np.concatenate((data, aug_data), axis=0)
    # augmented_labels = np.concatenate((labels, labels.repeat(multiplier)), axis=0)
    # return augmented_data, augmented_labels
    return aug_data, aug_labels


# 59通道 CR
def augment_with_CR(X, y):
    # 定义通道索引
    left_channel = np.array([2,4,6,9,11,13,15,18,20,22,24,27,29,31,33,35,37,39,41,44,46,48,51,53,55,58]) - 1
    right_channel = np.array([3,5,7,10,12,14,16,19,21,23,25,28,30,32,34,36,38,40,42,45,47,49,52,54,56,59]) - 1
    middle_channel = np.array([1,8,17,26,43,50,57]) - 1

    # # 排除 excluded_channels
    # left_channel = np.setdiff1d(left_channel, excluded_channels)
    # right_channel = np.setdiff1d(right_channel, excluded_channels)
    # middle_channel = np.setdiff1d(middle_channel, excluded_channels)

    X_aug = X.copy()
    y_aug = y.copy()

    # 交换左右通道的数据
    X_aug[:, left_channel, :] = X[:, right_channel, :]
    X_aug[:, right_channel, :] = X[:, left_channel, :]

    # label 0,1,2 分别对应左手, 右手, 双脚
    # 交换左右手的标签
    left_hand_indices = y == 0
    right_hand_indices = y == 1

    y_aug[left_hand_indices] = 1
    y_aug[right_hand_indices] = 0

    # 如果标签是双脚 2，标签不变
    return X_aug, y_aug


import pywt
import random
import numpy as np
from itertools import combinations

def DWTA(Xs, X_tar_train):
    wavename = 'db5'
    TcA, TcD = pywt.dwt(X_tar_train, wavename)
    ScA, ScD = pywt.dwt(Xs, wavename)
    Xs_aug = pywt.idwt(ScA, TcD, wavename, 'smooth')  # approximated component
    Xt_aug = pywt.idwt(TcA, ScD, wavename, 'smooth')  # approximated component
    return Xs_aug, Xt_aug


# NOTE: 只返回增强后的数据
def use_DWTA(data, labels, multiplier: int):
    if multiplier < 1:
        raise ValueError("multiplier must be at least 1")

    unique_labels = np.unique(labels)
    X_aug_list = list(data)  # 包含原始数据
    y_aug_list = list(labels)  # 包含原始标签
    
    for label in unique_labels:
        class_indices = np.where(labels == label)[0]
        X_class = data[class_indices]
        
        # 计算每个类别需要增强的样本数
        samples_to_generate = len(X_class) * (multiplier - 1)
        
        # 生成所有可能的样本对
        all_combinations = list(combinations(range(len(X_class)), 2))
        
        # 如果可能的组合数少于需要生成的样本数，就重复使用组合
        if len(all_combinations) < samples_to_generate:
            all_combinations = all_combinations * (samples_to_generate // len(all_combinations) + 1)
        
        # 随机选择需要的组合数
        selected_combinations = random.sample(all_combinations, samples_to_generate)
        
        for i, j in selected_combinations:
            Xs = X_class[i]
            X_tar_train = X_class[j]
            Xs_aug, Xt_aug = DWTA(Xs, X_tar_train)
            
            # 随机选择一个增强样本添加到列表中
            if random.choice([True, False]):
                X_aug_list.append(Xs_aug)
            else:
                X_aug_list.append(Xt_aug)
            y_aug_list.append(label)
    
    X_aug = np.array(X_aug_list)
    y_aug = np.array(y_aug_list)
    
    return X_aug, y_aug