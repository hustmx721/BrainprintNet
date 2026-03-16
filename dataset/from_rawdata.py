""" 
instruction:get raw data and process the data 
Author:hust-marx2
time: 2023/9/22
lastest:
"""
# add the father dir root to search the right functional file
import sys
sys.path.append('..')

import os
import scipy.io as scio
import numpy as np
from itertools import repeat, chain

# 命令行下载网站数据
# http://www.bnci-horizon-2020.eu/database/data-sets/004-2014/B04T.mat
def downloadfile():
    file_path = os.path.split(os.path.realpath(__file__))[0]
    os.chdir(file_path)
    os.system("cd BNCI2014004")
    file_list = [i+1 for i in range(0,9)]
    for idx in file_list:
        train_data_root = f"./B0{idx}T.mat"
        eval_data_root = f"./B0{idx}E.mat"
        os.system("mv"+" "+train_data_root+ " " +"./BNCI2014004")
        os.system("mv"+" "+eval_data_root+ " " +"./BNCI2014004")
##### 超参 #####
# ids = [1,2,3,4,5,6,7,8,9] 
# MI_time = 4
# dataroot = "/data2/tyl/data/BNCI2014001"

# 二分类数据集，左右手
# 六通道数据,前三通道EEG后三通道EOG,做运动想象分类,只能用EEG信号
# !注意trigger号开始的位置并不是运动想象开始的位置
def get_source_EEG_BNCI2014004(dataroot, id, split:str="train", MI_time:int=3):
    if split == "train":
        # 对原始数据进行处理，训练集数据
        raweeg = scio.loadmat(dataroot + f"B0{id}T.mat")["data"] # shape(1,3) three sessions:120 + 120 + 160 trials
        sessions_num = raweeg.shape[1]
        fs = raweeg[0][0]["fs"][0][0][0][0] # 250hz
        t_data = []
        t_label = []

        for session_idx in range(sessions_num):
            slice_data = []
            data = raweeg[0][session_idx]["X"][0][0].T # (channels,samples)
            trigger = raweeg[0][session_idx]["trial"][0][0].T.squeeze(axis=0) # trigger
            label = raweeg[0][session_idx]["y"][0][0].T.squeeze(axis=0) - 1 # 0为左手，1为右手
            trial_num = len(trigger) 
            # channel selection,remove the EOG signal
            data = data[0:3, :] # (120,6,750) -> (120,3,750)
            for idx in range(trial_num): 
                # mat index begin from 1 while python from 0
                # trial tigger begin from fixation cross
                position = int(trigger[idx] - 1 + fs * 3.5)
                slice_data.append(data[:, position : position + fs * MI_time]) 
            if session_idx == 0:
                t_data.append(slice_data)
                t_label.append(label)
                t_data = np.array(t_data).squeeze(axis=0)
                t_label = np.array(t_label).squeeze(axis=0)
            else:
                t_data = np.concatenate((t_data , np.array(slice_data)))
                t_label = np.concatenate((t_label , np.array(label))) 
        # 数据分类
        MI_left, MI_right = [] , []
        for idx,trial in enumerate(t_data):
            if t_label[idx] == 0:
                MI_left.append(trial)
            elif t_label[idx] == 1:
                MI_right.append(trial)

        # 训练集和验证集划分 400trails4:1划分训练集和验证集，训练集、测试集有320个trials，验证集80个
        ratio = 4 / 5
        trial_num = len(MI_left)
        # 拼接
        train_data = np.concatenate((np.array(MI_left)[:int(trial_num*ratio),:,:],np.array(MI_right)[:int(trial_num*ratio),:,:]),axis=0)
        train_label = np.concatenate((np.zeros(int(trial_num*ratio)),np.ones(int(trial_num*ratio))),axis=0)
        validate_data = np.concatenate((np.array(MI_left)[int(trial_num*ratio):,:,:],np.array(MI_right)[int(trial_num*ratio):,:,:]),axis=0)
        validate_label = np.concatenate((np.zeros(trial_num-int(trial_num*ratio)),np.ones(trial_num-int(trial_num*ratio))),axis=0)

        return train_data, train_label, validate_data, validate_label

    elif split == "test":
        raweeg = scio.loadmat(dataroot + f"B0{id}E.mat")["data"] # shape(1,2) two sessions:160 + 160 trials
        sessions_num = raweeg.shape[1]
        e_data = []
        e_label = []
        fs = raweeg[0][0]["fs"][0][0][0][0]
        for session_idx in range(sessions_num):
            slice_data = []
            data = raweeg[0][session_idx]["X"][0][0].T # (channels,samples)
            trigger = raweeg[0][session_idx]["trial"][0][0].T.squeeze(axis=0) 
            label = raweeg[0][session_idx]["y"][0][0].T.squeeze(axis=0) - 1
            trial_num = len(trigger) 
            # channel selection,remove the EOG signal
            data = data[0:3, :]
            for idx in range(trial_num):
                position = int(trigger[idx] - 1 + fs * 3.5)
                slice_data.append(data[:, position : position + fs * MI_time]) # (120,6,1000)
            if session_idx == 0:
                e_data.append(slice_data)
                e_label.append(label)
                e_data = np.array(e_data).squeeze(axis=0)
                e_label = np.array(e_label).squeeze(axis=0)
            else:
                e_data = np.concatenate((e_data , np.array(slice_data)))
                e_label = np.concatenate((e_label , np.array(label)))  
  
        return e_data,e_label

    else: 
        raise NotImplementedError

# 四分类数据集，左右手、双脚和舌头
def get_source_EEG_BNCI2014001(dataroot, id, split:str="train", MI_time:int=3):
    if split == "train":
        raweeg = scio.loadmat(os.path.join(dataroot,f"A0{id}T.mat")) 
        # raweeg["data"].shape= (1,9),前三个session没有tigger和标签,288=6*48
        # 注意04T只有7个cell,其余有9个cell
        if id == 4 :
            file_list = [1,2,3,4,5,6] # A04T只有7个cell
        else :
            file_list = [3,4,5,6,7,8] # 后6个session才有数据
        """ 
        # print(raweeg["data"][0][3]["X"][0][0].shape) # (96735, 25)
        # print(raweeg["data"][0][3]["trial"][0][0].shape) # (48,1)
        # print(raweeg["data"][0][3]["y"][0][0].shape) # (48,1)
        # print(raweeg["data"][0][0]["fs"][0][0][0][0]) # 250Hz
        # print(raweeg["data"][0][0]["classes"][0][0][0][0-3]) # left hand, right hand, feet, tongue
        """
        fs = raweeg["data"][0][0]["fs"][0][0][0][0] # 250Hz
        t_data = []
        t_label = []
        for session_idx in file_list:
            data = raweeg["data"][0][session_idx]["X"][0][0].T # (channels,points)
            label = raweeg["data"][0][session_idx]["y"][0][0].squeeze() - 1 
            trigger = raweeg["data"][0][session_idx]["trial"][0][0].squeeze()
            trial_num = len(trigger)
            # channel selection:the last three channels were EOG siganls
            data = np.array(data[:-3, :])
            one_session_data = np.zeros((trial_num,data.shape[0],fs * MI_time))
            # get one session data
            for idx in range(trial_num):
                position = int(trigger[idx] -1 + fs * 2.5)
                one_session_data[idx,:,:] = data[:, position : position + MI_time * fs]
            t_data.append(one_session_data)
            t_label.append(label)
            # all train data:(288,22,750)
            if session_idx ==8: 
                t_data = np.array(t_data).reshape((-1,data.shape[0],fs * MI_time))
                t_label = np.array(t_label).reshape(-1)
            if file_list[0] == 1 and session_idx == 6:
                t_data = np.array(t_data).reshape((-1,data.shape[0],fs * MI_time))
                t_label = np.array(t_label).reshape(-1)

        # 为了保证训练集和验证集类别平衡,分别对每类进行划分
        MI_left, MI_right, MI_feet, MI_tongue = [], [], [], []
        for idx,trial in enumerate(t_data):
            if t_label[idx] == 0:
                MI_left.append(trial)
            elif t_label[idx] == 1 :
                MI_right.append(trial)  
            elif t_label[idx] == 2:
                MI_feet.append(trial)
            elif t_label[idx] == 3 :
                MI_tongue.append(trial) 

        trial_num = t_label.shape[0] / 4 # 288 */4 = 72
        ratio = 5/6 # 72*5/6 = 60

        train_data = np.concatenate((np.array(MI_left)[:int(trial_num*ratio),:,:],
                                     np.array(MI_right)[:int(trial_num*ratio),:,:],
                                     np.array(MI_feet)[:int(trial_num*ratio),:,:],
                                     np.array(MI_tongue)[:int(trial_num*ratio),:,:]),
                                     axis=0)
        validate_data = np.concatenate((np.array(MI_left)[int(trial_num*ratio):,:,:],
                                     np.array(MI_right)[int(trial_num*ratio):,:,:],
                                     np.array(MI_feet)[int(trial_num*ratio):,:,:],
                                     np.array(MI_tongue)[int(trial_num*ratio):,:,:]),
                                     axis=0)
        train_label =  np.array(list(chain.from_iterable(repeat(x, int(trial_num * ratio)) for x in range(4)))) # (240,22,750)
        validate_label = np.array(list(chain.from_iterable(repeat(x, int(trial_num - trial_num*ratio)) for x in range(4)))) # (48,22,750)

        return train_data, train_label, validate_data, validate_label
    
    elif split == "test":
        raweeg = scio.loadmat(os.path.join(dataroot,f"A0{id}E.mat")) 
        file_list = [3,4,5,6,7,8] # 后6个session才有数据
        fs = raweeg["data"][0][0]["fs"][0][0][0][0] # 250Hz
        e_data = []
        e_label = []
        for session_idx in file_list:
            data = raweeg["data"][0][session_idx]["X"][0][0].T # (channels,points)
            label = raweeg["data"][0][session_idx]["y"][0][0].squeeze() - 1 
            trigger = raweeg["data"][0][session_idx]["trial"][0][0].squeeze()
            trial_num = len(trigger)
            # channel selection:the last three channels were EOG siganls
            data = np.array(data[:-3, :])
            one_session_data = np.zeros((trial_num,data.shape[0],fs * MI_time))
            # get one session data
            for idx in range(trial_num):
                position = int(trigger[idx] -1 + fs * 2.5)
                one_session_data[idx,:,:] = data[:, position : position + MI_time * fs]
            e_data.append(one_session_data)
            e_label.append(label)
            # all train data:(288,22,750)
            if session_idx ==8: 
                e_data = np.array(np.array(e_data).reshape((-1,data.shape[0],fs * MI_time)))
                e_label = np.array(np.array(e_label).reshape(-1))
            
        return e_data, e_label

    else: 
        raise NotImplementedError


