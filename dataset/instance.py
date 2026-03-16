""" 
instruction:get dataset instance
Author:hust-marx2
time: 2023/9/22
lastest:
"""
import torch
import random
import numpy as np
import os
from torch.utils.data import Dataset,DataLoader

# 随机种子初始化,保证相同的种子实验可重复性
def set_seed(myseed = 23721):
    torch.set_num_threads(1) # num of threads
    random.seed(myseed) # generate random seed
    torch.manual_seed(myseed) # the same seed of pytorch for CPU
    torch.cuda.manual_seed(myseed) # the same seed of pytorch for GPU
    torch.cuda.manual_seed_all(myseed) # if you are using multi-GPU
    np.random.seed(myseed) # numpy also the same seed
    os.environ["PYTHONSEED"] = str(myseed) # back state_dict value to set const hash
    torch.backends.cudnn.enabled = False # forbidden to use unsure algorithm
    torch.backends.cudnn.deterministic = True # use the same conv

# 生成数据集   
class MyDataset(Dataset):
    def __init__(self,data,label):
        self.data = data
        self.label = label
    
    def __getitem__(self, idx):
        data = self.data[idx].astype(np.float32)
        label = self.label[idx]
        return data, label
    
    def __len__(self):
        return self.data.shape[0]

# 生成dataloader
def trans_torch(data, label, mode): # "train"/"test"
    # 001/004 : bs=256; other : bs=512
    data_set = MyDataset(data,label)
    if mode == "train":
        dataloader = DataLoader(dataset = data_set, batch_size = 64, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    elif mode == "test":
        dataloader = DataLoader(dataset = data_set, batch_size = 64, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    return dataloader