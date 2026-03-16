import os
import torch.nn as nn
import sys
import errno
import shutil
import os.path as osp
import argparse
import torch
import scipy

from typing import Tuple, List

sys.path.append("/mnt/data1/tyl/UserID")

from models.MyModels import EEGNet,ShallowConvNet,DeepConvNet,CNN_LSTM
from models.EEGConformer import Conformer
from models.FBCNet import FBCNet
from models.FBMSNet import FBMSNet
from models.constraintmodels import MyConformerWithConstraint
from models.modelV1 import ResEEGNet
from models.ResEEGNet import ResEEGNet_v1, ResEEGNet_v2
from models.IFNet import IFNet
from models.FBResEEGNet import FBResEEGNet_v1
from models.FullConv import FullConv
from models.FBMoE import MoETSformer
from models.FBMoE1 import MoETSformer1
from models.FBMoE2 import MoETSformer2
from models.FBMoE3 import MoETS
from models.FBResEEG2D import FBResEEG2D
from models.FBResEEG2D4 import FBResEEG2D4
from models.FBResEEGMoE import FBResEEGMoE
from models.FBCNet2 import FBCNet2
from models.FBResEEGNet3 import FBResEEGNet3
from models.FBResEEGNet4 import FBResEEGNet4
from models.FBResEEGNet5 import FBResEEGNet5
from models.FBResEEGNet6 import FBResEEGNet6
from models.DBNet import DBNet
from models.DBNet1 import DBNet1
from models.DBNet2 import DBNet2
from models.DBNet3 import DBNet3, DBNet3_1
from models.trybest import trybest_4hz, trybest_6hz, trybest_8hz, trybest_12hz, trybest_24hz, trybest_2hz1
from models.TSAPNet import TSAPNet
from models.lastmodel import lastmodel
from models.best import best
from models.best2models import FBMSTSNet, MSNet, FBMSTSNet_FeatureExtractor, FBMSTSNet_Classifier, MSNet_FeatureExtractor, MSNet_Classifier
from models.ablation_models import Net_1_x_3
from baseline.frameworks.DARN.my_network import DARNNet
from baseline.frameworks.DRFNet.my_network import DRFNet
from baseline.frameworks.MCGP.MCGP import MCGPnet



from handifea.fea import *
from baseline.dataloader import *

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

# 确保argsparse的输出为bool变量
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True','t', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False','f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))


def load_baseline_model(args:argparse.ArgumentParser): 
    device = torch.device("cuda:"+str(args.gpuid) if torch.cuda.is_available() else "cpu")
    # dynamic tuning the lr -- MultiStepLr
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,100,150],gamma=0.25)
    torch.cuda.empty_cache()
    torch.cuda.set_device(device)
    # EEGNet, DeepConvNet, ShallowConvNet, Conformer, FBCNet, MyConformerWithConstraint, ResEEGNet, 1D_LSTM
    if args.model == "EEGNet":
        model = EEGNet(classes_num=args.nclass,Chans=args.channel,Samples=int(args.fs*args.timepoint)).to(device)
    elif args.model == "DeepConvNet":
        model = DeepConvNet(classes_num=args.nclass,Chans=args.channel,Samples=int(args.fs*args.timepoint)).to(device)
    elif args.model == "ShallowConvNet":
        model = ShallowConvNet(classes_num=args.nclass,Chans=args.channel,Samples=int(args.fs*args.timepoint)).to(device)
    elif args.model == "Conformer":
        model = Conformer(Chans=args.channel, Samples=int(args.fs*args.timepoint), n_classes=args.nclass).to(device)
    elif args.model == "FBCNet":
        model = FBCNet(nChan=args.channel,nClass=args.nclass,nBands=12, fs=args.fs).to(device)
    elif args.model == "FBCNet2":
        model = FBCNet2(nChan=args.channel,nClass=args.nclass,nBands=6, fs=args.fs, Samples=int(args.fs*args.timepoint)).to(device)
    elif args.model == "FBMSNet":
        model = FBMSNet(nChan=args.channel,nClass=args.nclass, fs=args.fs, nTime=int(args.fs*args.timepoint)).to(device)
    elif args.model == "IFNet":
        model = IFNet(in_planes=args.channel,out_planes=64,num_classes=args.nclass, kernel_size=63, 
                      radix=2, patch_size=int(args.fs//2), fs=args.fs, time_points=int(args.fs*args.timepoint)).to(device)
    elif args.model == "MyConformerWithConstraint":
        model = MyConformerWithConstraint(kernels=[11,21,31,41,51],n_classes=args.nclass,
                    Chans=args.channel, Samples=int(int(args.fs*args.timepoint)),
                    emb_size=64,depth=4).to(device)
    # self-define ResEEGNet, all kinds of versions
    elif args.model == "ResEEGNet":
        model = ResEEGNet(kernels=[11,21,31,41,51],num_classes=args.nclass,
                          in_channels=args.channel,Samples=int(args.fs*args.timepoint),
                          Resblock=args.resblock).to(device)
    elif args.model == "FullConv":
        model = FullConv(kernels=[7,15,31,63,127],num_classes=args.nclass,
                          in_channels=args.channel,Samples=int(args.fs*args.timepoint),
                          Resblock=args.resblock).to(device)
    elif args.model == "MoETSformer":
        model = MoETSformer(kernels=[7,15,31,63,127], nbands=6, n_classes=args.nclass, emb_size=64,
                          Chans=args.channel, fs=args.fs, num_experts=8, depth=4).to(device)
    elif args.model == "MoETSformer1":
        model = MoETSformer1(kernels=[7,15,31,63,127], nbands=6, n_classes=args.nclass, emb_size=64, m=8,
                          Chans=args.channel, fs=args.fs, num_experts=6, depth=1).to(device)
    elif args.model == "MoETSformer2":
        model = MoETSformer2(kernels=[11,21,31,41,51], nbands=6, n_classes=args.nclass, emb_size=64, m=8,
                          Chans=args.channel, fs=args.fs, num_experts=6, depth=1).to(device)
    elif args.model == "MoETS":
        model = MoETS(kernels=[11,21,31,41,51], nbands=6, n_classes=args.nclass, emb_size=64, m=8,
                          Chans=args.channel, fs=args.fs, num_experts=6, depth=1).to(device)
    elif args.model == "FBResEEGNet_v2":
        model = FBResEEGNet_v1(kernels=[11,21,31,41,51],num_classes=args.nclass,
                          in_channels=args.channel,Samples=int(args.fs*args.timepoint),
                          Resblock=args.resblock, hidden_chans=64).to(device)
    elif args.model == "ResEEGNet_v1":
        model = ResEEGNet_v1(kernels=[11,21,31,41,51],num_classes=args.nclass,
                          in_channels=args.channel,Samples=int(args.fs*args.timepoint),
                          Resblock=5).to(device)
    elif args.model == "ResEEGNet_v2":
        model = ResEEGNet_v2(kernels=[11,21,31,41,51],num_classes=args.nclass,
                          in_channels=args.channel,Samples=int(args.fs*args.timepoint),
                          Resblock=5).to(device)
    elif args.model == "FBResEEG2D":
        model = FBResEEG2D(kernels=[11,21,31,41,51],num_classes=args.nclass,
                          in_channels=args.channel,Samples=int(args.fs*args.timepoint),
                          Resblock=5).to(device)
    elif args.model == "FBResEEG2D4":
        model = FBResEEG2D4(kernels=[11,21,31,41,51],num_classes=args.nclass, fs=args.fs,
                          in_channels=args.channel,Samples=int(args.fs*args.timepoint),
                          Resblock=5, nbands=6).to(device)
    elif args.model == "FBResEEGMoE":
        model = FBResEEGMoE(kernels=[11,21,31,41,51],num_classes=args.nclass, fs=args.fs,
                          in_channels=args.channel,Samples=int(args.fs*args.timepoint),
                          nbands=6).to(device)
    elif args.model == "1D_LSTM":
        model = CNN_LSTM(channels=args.channel, n_classes=args.nclass, time_points=int(args.timepoint*args.fs), 
                         hidden_size=128, num_layers=2).to(device)
    elif args.model == "FBResEEGNet3":
        model = FBResEEGNet3(kernels=[11,21,31,41,51],num_classes=args.nclass, strideFactor=4,
                          in_channels=args.channel,Samples=int(args.fs*args.timepoint),
                          Resblock=args.resblock, hidden_chans=64).to(device)
    elif args.model == "FBResEEGNet4":
        model = FBResEEGNet4(kernels=[11,21,31,41,51],num_classes=args.nclass, strideFactor=16,
                          in_channels=args.channel,Samples=int(args.fs*args.timepoint),
                          Resblock=args.resblock, hidden_chans=64).to(device)
    elif args.model == "FBResEEGNet5":
        model = FBResEEGNet5(kernels=[11,21,31,41,51],num_classes=args.nclass, strideFactor=4,
                          in_channels=args.channel,Samples=int(args.fs*args.timepoint),
                          Resblock=2, hidden_chans=64).to(device)
    elif args.model == "FBResEEGNet6":
        model = FBResEEGNet6(kernels=[11,21,31,41,51],num_classes=args.nclass, strideFactor=4, m=8,
                          in_channels=args.channel,Samples=int(args.fs*args.timepoint),
                          Resblock=2, hidden_chans=64).to(device)
    elif args.model == "DBNet":
        model = DBNet(kernels=[11,21,31,41,51],num_classes=args.nclass, strideFactor=4,
                          in_channels=args.channel,Samples=int(args.fs*args.timepoint),
                          hidden_chans=64, emb_size=64).to(device)
    elif args.model == "DBNet1":
        model = DBNet1(kernels=[11,21,31,41,51],num_classes=args.nclass, strideFactor=4,
                          in_channels=args.channel,Samples=int(args.fs*args.timepoint),
                          hidden_chans=64).to(device)
    elif args.model == "DBNet2":
        model = DBNet2(kernels=[11,21,31,41,51],num_classes=args.nclass, strideFactor=5,
                          in_channels=args.channel,Samples=int(args.fs*args.timepoint),
                          hidden_chans=64).to(device)
    elif args.model == "DBNet3":
        model = DBNet3(kernels=[11,21,31,41,51],num_classes=args.nclass, strideFactor=5,
                          in_channels=args.channel,Samples=int(args.fs*args.timepoint),
                          hidden_chans=64).to(device)
    elif args.model == "DBNet3_1":
        model = DBNet3_1(kernels=[11,21,31,41,51],num_classes=args.nclass, strideFactor=4,
                          in_channels=args.channel,Samples=int(args.fs*args.timepoint),
                          hidden_chans=64).to(device)
    elif args.model == "trybest_4hz":
        model = trybest_4hz(kernels=[11,21,31,41,51],num_classes=args.nclass, strideFactor=4,
                          in_channels=args.channel, nbands=6).to(device)
    elif args.model == "trybest_6hz":
        model = trybest_6hz(kernels=[11,21,31,41,51],num_classes=args.nclass, strideFactor=4,
                          in_channels=args.channel, nbands=4).to(device)
    elif args.model == "trybest_8hz":
        model = trybest_8hz(kernels=[11,21,31,41,51],num_classes=args.nclass, strideFactor=4,
                          in_channels=args.channel, nbands=3).to(device)
    elif args.model == "trybest_12hz":
        model = trybest_12hz(kernels=[11,21,31,41,51],num_classes=args.nclass, strideFactor=4,
                          in_channels=args.channel, nbands=2).to(device)
    elif args.model == "trybest_24hz":
        model = trybest_24hz(kernels=[11,21,31,41,51],num_classes=args.nclass, strideFactor=4,
                          in_channels=args.channel, nbands=1).to(device)
    elif args.model == "trybest_2hz1":
        model = trybest_2hz1(kernels=[11,21,31,41,51],num_classes=args.nclass, strideFactor=4,
                          in_channels=args.channel, nbands=12).to(device)
    elif args.model == "TSAPNet":
        model = TSAPNet(kernels=[11,21,31,41,51],Samples=int(args.fs*args.timepoint),
                        num_classes=args.nclass, strideFactor=4, fs=args.fs,
                          in_channels=args.channel, nbands=12).to(device)
    elif args.model == "lastmodel":
        model = lastmodel(kernels=[11,21,31,41,51],
                        num_classes=args.nclass, strideFactor=4, 
                          in_channels=args.channel, nbands=12).to(device)
    elif args.model == "best":
        model = best(kernels=[11,21,31,41,51],
                        num_classes=args.nclass, strideFactor=args.strideFactor, 
                          in_channels=args.channel, nbands=12).to(device)
    elif args.model == "FBMSTSNet":
        model = FBMSTSNet(kernels=[7, 15, 31, 63, 127], fs=args.fs,
                        num_classes=args.nclass,
                          in_channels=args.channel, nbands=12).to(device)
        if args.tl is not None:
            netF = FBMSTSNet_FeatureExtractor(kernels=[11,21,31,41,51], fs=args.fs,
                        num_classes=args.nclass, 
                          in_channels=args.channel, nbands=12).to(device)
            netC = FBMSTSNet_Classifier(kernels=[11,21,31,41,51], fs=args.fs,
                        num_classes=args.nclass, 
                          in_channels=args.channel, nbands=12).to(device)
            model = [netF, netC]
    elif args.model == "MSNet":
        model = MSNet(kernels=[11,21,31,41,51],
                        num_classes=args.nclass,
                          in_channels=args.channel, hidden_chans=64).to(device)
        if args.tl is not None:
            netF = MSNet_FeatureExtractor(kernels=[11,21,31,41,51],
                        num_classes=args.nclass,
                          in_channels=args.channel, hidden_chans=64).to(device)
            netC = MSNet_Classifier(kernels=[11,21,31,41,51],
                        num_classes=args.nclass,
                          in_channels=args.channel, hidden_chans=64).to(device)
            model = [netF, netC]
    # ablation models
    elif args.model == "Net_1_x_3":
        model = Net_1_x_3(kernels=[31], fs=args.fs,
                        num_classes=args.nclass, 
                          in_channels=args.channel, nbands=12).to(device)
    elif args.model == "DARNNet":
        model = DARNNet(device=device, input_size=(1, args.channel, int(args.fs*args.timepoint)),
                        num_class=args.nclass).to(device)
    elif args.model == "DRFNet":
        model = DRFNet(device=device, input_size=(1, args.channel, int(args.fs*args.timepoint)),
                        num_class=args.nclass).to(device)
    elif args.model == "MCGPnet":
        model = MCGPnet(device=device, num_class=args.nclass, input_size=(1, args.channel, int(args.fs*args.timepoint)),
                        sampling_rate= int(args.fs*args.timepoint)).to(device)
    else:
        raise ValueError("Invalid model name")
    
    # optimizer = torch.optim.Adam(nn.Sequential(*model).parameters(), lr=args.initlr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initlr)

    return device, model, optimizer

def load_data(args:argparse.ArgumentParser):
    # "001","004","BCI85","Rest85", "LJ30", OpenBMI:"Rest", "MI", "ERP", "SSVEP", "cross"
    OpenBMI = ["Rest", "MI", "ERP", "SSVEP", "CatERP"]
    M3CV = ["M3CV_Rest", "M3CV_Transient", "M3CV_Steady", "M3CV_P300", "M3CV_Motor", "M3CV_SSVEP_SA"]
    if args.setsplit in ["001", "004"]:
        trainloader, valloader, testloader = GetLoader14xxx(args.seed,args.setsplit)
    elif args.setsplit == "BCI85":
        trainloader, valloader, testloader = GetloaderBCI85(args.seed)
    elif args.setsplit == "Rest85":
        trainloader, valloader, testloader = GetloaderRest85(args.seed)
    elif args.setsplit == "LJ30":
        trainloader, valloader, testloader = GetloaderLJ30(args.seed, aug_type=args.aug_type)
    elif args.setsplit in OpenBMI:
        trainloader, valloader, testloader = GetLoaderOpenBMI(args.seed,split=args.setsplit, aug_type=args.aug_type)
    elif args.setsplit in M3CV:
        trainloader, valloader, testloader = GetLoaderM3CV(args.seed,split=args.setsplit, aug_type=args.aug_type)
    elif args.setsplit == "SEED":
        trainloader, valloader, testloader = GetLoaderSEED(args.seed)
    elif args.setsplit is None and args.cross_task is not None and set(args.cross_task).issubset(set(M3CV)) :
        trainloader, valloader, testloader = M3CV_crosstask(seed=args.seed, split=args.cross_task, cross_session=False, session_num=args.session_num)
    elif args.setsplit is None and args.cross_task is not None and set(args.cross_task).issubset(set(OpenBMI)):
        trainloader, valloader, testloader = OpenBMI_crosstask(seed=args.seed, split=args.cross_task, cross_session=True, session_num=args.session_num)
    else:
        raise ValueError("Invalid setsplit name")

    return trainloader, valloader, testloader

def set_args(args:argparse.ArgumentParser):
    OpenBMI = ["Rest", "MI", "SSVEP", "cross", "CatERP", "ERP"]
    M3CV = ["M3CV_Rest", "M3CV_Transient", "M3CV_Steady", "M3CV_P300", "M3CV_Motor", "M3CV_SSVEP_SA"]
    # TODO 暂时跨范式的参数未定
    if args.setsplit == "001":
        args.nclass = 9
        args.channel = 22
        args.fs = 250
        args.timepoint = 4
    elif args.setsplit == "004":
        args.nclass = 9
        args.channel = 3
        args.fs = 250
        args.timepoint = 4
    elif args.setsplit in ["BCI85", "Rest85"]:
        args.nclass = 85
        args.channel = 27
        args.fs = 250
        args.timepoint = 5
    elif args.setsplit == "LJ30":
        args.nclass = 30
        args.channel = 20
        args.fs = 300
        args.timepoint = 5
    elif args.setsplit in OpenBMI:
        args.nclass = 54
        args.channel = 62
        args.fs = 250
        args.timepoint = 4
    elif args.setsplit == "ERP":
        args.nclass = 54
        args.channel = 62
        args.fs = 1000
        args.timepoint = 0.8
    elif args.setsplit in M3CV:
        args.nclass = 20
        args.channel = 64
        args.fs = 250
        args.timepoint = 4
    elif args.setsplit == "SEED":
        args.nclass = 15
        args.channel = 62
        args.fs = 200
        args.timepoint = 4
    # M3CV 跨任务
    elif args.setsplit is None and args.cross_task is not None and set(args.cross_task).issubset(set(M3CV)):
        args.nclass = 20
        args.channel = 64
        args.fs = 250
        args.timepoint = 4
    # OpenBMI 跨任务
    elif args.setsplit is None and args.cross_task is not None and set(args.cross_task).issubset(set(OpenBMI)):
        if "ERP" not in args.cross_task:
            args.nclass = 54
            args.channel = 62
            args.fs = 250
            args.timepoint = 4
        else:
            args.nclass = 54
            args.channel = 62
            args.fs = 1000
            args.timepoint = 0.8
    return args

def get_subband_fea(data,csp,f_start=8, f_end=32, bandwidth=4, fs=250):
    # (N,C,T) -> (N,nBands,C,T)
    data = SubBandSplit(data,freq_start=f_start,freq_end=f_end,bandwidth=bandwidth,fs=250).transpose((0,3,1,2))
    n_bands = (f_end - f_start) // bandwidth 
    fea = []
    for sub_band in range(n_bands):
        wavelet_fea = WaveletPacket(data[:,sub_band,:,:])
        PSD_fea, _ = PSD(data[:,sub_band,:,:],{"stftn": 512,"fStart": [1,4,8,14,31],"fStop": [3,7,13,30,50],"fs": fs, "EyeTime":1})
        AR_fea = AR_burg(data[:,sub_band,:,:])
        CSP_fea = csp.transform(data[:,sub_band,:,:])
        [wavelet_fea, PSD_fea, AR_fea, CSP_fea] = [fea.reshape((fea.shape[0],-1)) for fea in [wavelet_fea, PSD_fea, AR_fea, CSP_fea]]
        subband_fea = np.concatenate((wavelet_fea, PSD_fea, AR_fea, CSP_fea), axis=-1)
        fea.append(subband_fea)
    print(wavelet_fea.shape, PSD_fea.shape, AR_fea.shape, CSP_fea.shape, subband_fea.shape)
    fea = np.concatenate(fea,axis=-1)
    return fea

def get_fea(data,csp):
    # (N,C,T)
    wavelet_fea = WaveletPacket(data)
    PSD_fea, _ = PSD(data,{"stftn": 512,"fStart": [1,4,8,14,31],"fStop": [3,7,13,30,50],"fs": 250, "EyeTime":1})
    AR_fea = AR_burg(data)
    CSP_fea = csp.transform(data)
    [wavelet_fea, PSD_fea, AR_fea, CSP_fea] = [fea.reshape((fea.shape[0],-1)) for fea in [wavelet_fea, PSD_fea, AR_fea, CSP_fea]]
    fea = np.concatenate((wavelet_fea, PSD_fea, AR_fea, CSP_fea), axis=-1)
    print(wavelet_fea.shape, PSD_fea.shape, AR_fea.shape, CSP_fea.shape, fea.shape)
    return fea


# 正则化损失项
def L1_norm(model:nn.Module, lamda:float=1e-4):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lamda * l1_norm


def L2_norm(model:nn.Module, lamda:float=1e-4):
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    return lamda * l2_norm


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter)**(-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


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
    sub_band_data = np.stack(results, axis=-1)

    return rearrange(sub_band_data, 'b c t n -> b (c n) t')



def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    splits: List[float],
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    将数据集按照指定的比例进行分层划分。

    参数:
    X : np.ndarray
        特征数据
    y : np.ndarray
        标签数据
    splits : List[float]
        划分比例,必须和为1
    random_state : int, 可选
        随机种子,用于reproducibility

    返回:
    List[Tuple[np.ndarray, np.ndarray]]
        包含划分后的(X, y)对的列表
    """
    if abs(sum(splits) - 1.0) > 1e-6:
        raise ValueError("划分比例之和必须为1")

    np.random.seed(random_state)
    classes = np.unique(y)
    n_splits = len(splits)
    result = [[] for _ in range(n_splits)]

    for c in classes:
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        split_points = np.cumsum([int(len(idx) * split) for split in splits[:-1]])
        split_idx = np.split(idx, split_points)
        
        for i, indices in enumerate(split_idx):
            result[i].extend(indices)

    result = [(X[idx], y[idx]) for idx in result]
    return result