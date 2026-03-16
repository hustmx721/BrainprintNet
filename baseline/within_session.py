
import argparse
import os
import torch
import sys
import time
import wandb
import gc
sys.path.append("/mnt/data1/tyl/UserID/")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torchinfo import summary
from tabulate import tabulate
# from torchstat import stat
# stat(model,input_size)

from baseline.dataloader import *
from dataset.instance import set_seed
from evaluate import evaluate, evaluate_centerloss
from models.modelV1 import ResEEGNet
from utils.utlis import load_baseline_model, load_data, set_args, str2bool
from utils.loss import CenterLoss
from utils.mylogging import Logger
from utils.feature_vis import FeatureVisualize
from utils.Attack import AWP, FGSM
from tlutils.loss import CELabelSmooth
from dataloader import SubBandSplit


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

# split = "M3CV_Rest"， "M3CV_Transient", "M3CV_Steady", "M3CV_P300", "M3CV_Motor", "M3CV_SSVEP_SA"
def GetLoaderWithinSessionM3CV(seed, split:str="M3CV_Rest"): 
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
    data, label = np.concatenate([train_x, test_x]), np.concatenate([train_s, test_s])
    # train, validation, test split
    tx, ex, ts, es = train_test_split(data, label, test_size=0.2, random_state=seed, stratify=label)
    tx, vx, ts, vs = train_test_split(tx, ts, test_size=0.2, random_state=seed, stratify=ts)
    
    tx, vx, ex = [np.expand_dims(x, axis=1) for x in [tx, vx, ex]] # if EEGNet, DeepConvNet or ShallowConvNet
    print(f"数据比例-----训练集:验证集:测试集 = {tx.shape}:{vx.shape}:{ex.shape}")
    [trainloader, validateloader, testloader] = [trans_torch(x,y,mode) for x,y,mode \
                                                 in zip([tx, vx, ex], [ts, vs, es], ["train","test","test"])]
    del data1, data2, train_x, train_s, ex, es, tx, vx, ts, vs
    gc.collect()
    return trainloader, validateloader, testloader




def train(trainloader,valloader,savepath,args):
    print( "-" * 20 + "开始训练!" + "-" * 20)
    device, model, optimizer = load_baseline_model(args)
    torch.cuda.empty_cache()
    torch.cuda.set_device(device)
    # wandb.watch(model)
    # loss funcs
    clf_loss_func = nn.CrossEntropyLoss().to(device)
    # clf_loss_func = CELabelSmooth(num_classes=args.nclass).to(device)

    # 训练指标
    best_epoch = 0
    best_acc = 0 # in validate dataset
    train_acc_all = []
    val_acc_all = []
    loss_item_train = []
    loss_item_val = []

    for epoch in tqdm(range(args.epoch),desc="Training:"):
        train_correct = 0
        one_epoch_loss = 0

        for _,(b_x,b_y) in enumerate(trainloader):
            model.train()
            optimizer.zero_grad()

            b_x, b_y = b_x.to(device), b_y.to(device)
            features, output = model(b_x) # 输出每个类别的概率
            celoss = clf_loss_func(output,b_y.long())
            # celoss = clf_loss_func(output,F.one_hot(b_y.long(),num_classes=args.nclass).long()) # CELabelSmooth
  
            # # IMloss
            # softmax_out = torch.softmax(output,dim=-1)
            # msoftmax = softmax_out.mean(dim=0)
            # im_div_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5)).to(device)

            # loss = celoss + im_div_loss
            loss = celoss 

            one_epoch_loss += loss.item() # 一个epoch的loss累加
            loss.backward()

            optimizer.step()

            # FGSM attack
            """             
            awp_start = 0
            if epoch > awp_start:
                optimizer.zero_grad()
                perturbed_data = attack.attack(b_x.float(), b_y.long())
                fea_perturbed, out_perturbed = model(perturbed_data)
                attack_loss = clf_loss_func(out_perturbed, b_y.long())
                attack_loss.backward()
                optimizer.step() 
            """

            # scheduler.step()
            pred_cls = torch.argmax(output, dim=1)  # 置信度最高的类别
            train_correct += float((pred_cls == b_y).sum())
        one_epoch_loss_avg = one_epoch_loss / len(trainloader) # batch数
        loss_item_train.append(one_epoch_loss_avg)
        train_acc = train_correct / len(trainloader.dataset) # trial数
        train_acc_all.append(train_acc)

        # log loss and acc
        # wandb.log({"train_acc":train_acc, "train_loss":one_epoch_loss_avg})

        # 每个epoch的验证集
        with torch.no_grad():
            val_loss, val_acc, val_f1, val_eer = evaluate(model,valloader,args)
            # wandb.log({"val_acc":val_acc, "val_loss":val_loss})
        # 注释掉这两行,则不早停
        if (epoch - best_epoch) > args.earlystop: # 早停 
            break
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 
            os.path.join(savepath, f"WithinSession_{args.model}_{args.seed}.pth"))
        val_acc_all.append(val_acc)  # 验证集准确率list
        loss_item_val.append(val_loss)

        # 打印训练损失
        if (epoch + 1) % 10 == 0 :
            print(f"Epoch:{epoch+1}\tTrain_acc:{train_acc:.4f}\tVal_acc:{val_acc:.4f}\tTrain_loss:{one_epoch_loss_avg:.6f}\tVal_loss:{val_loss:.6f}")

    print( "-" * 20 + "训练完成!" + "-" * 20)
    print(f"总训练轮数-{epoch+1}, 早停轮数-{best_epoch+1}")
    print(f"验证集最佳准确率为{best_acc*100:.2f}%")
    print(f"验证集平均准确率为{np.mean(np.array(val_acc_all))*100:.2f}%")
    return model

def train_centerloss(trainloader,valloader,savepath,args):
    print( "-" * 20 + "开始训练!" + "-" * 20)
    # to set model in device (cpu/gpu)
    device = torch.device("cuda:"+str(args.gpuid) if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.cuda.set_device(device)
    model = ResEEGNet(kernels=[11,21,31,41,51],num_classes=args.nclass,
                          in_channels=args.channel,Samples=int(args.fs*args.timepoint),
                          Resblock=args.resblock).to(device)
    fea_dim = model._get_fea_dim(torch.randn(args.bs,args.channel,int(args.fs*args.timepoint)).to(device))[-1] 
    # wandb.watch(model)
    # print(model)
    # loss funcs
    clf_loss_func = nn.CrossEntropyLoss().to(device)
    ctr_loss_func = CenterLoss(num_classes=args.nclass,feat_dim=fea_dim,use_gpu=True)
    params = list(model.parameters()) + list(ctr_loss_func.parameters())
    optimizer = torch.optim.Adam(params, lr=args.initlr)
    # dynamic tuning the lr -- MultiStepLr
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,100,150],gamma=0.25)
    # 训练指标
    best_epoch = 0
    best_acc = 0 # in validate dataset
    train_acc_all = []
    val_acc_all = []
    loss_item_train = []
    loss_item_val = []

    for epoch in tqdm(range(args.epoch),desc="Training:"):
        train_correct = 0
        one_epoch_loss = 0
        for _,(b_x,b_y) in enumerate(trainloader):
            model.train()
            optimizer.zero_grad()

            b_x = b_x.to(device)
            b_y = b_y.to(device)
            features, output = model(b_x) # 输出每个类别的概率
            clf_loss = clf_loss_func(output,b_y.long())
            ctr_loss = ctr_loss_func(features,b_y.long())

            # softmax_out = torch.softmax(output,dim=-1)
            # msoftmax = softmax_out.mean(dim=0)
            # im_div_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5)).to(device)
            # print(f"epoch{epoch+1}:")
            # print(f"clf_loss:{clf_loss.item():.6f}", f"ctr_loss:{ctr_loss.item():.6f}", f"im_div_loss:{im_div_loss.item():.6f}")

            # loss = clf_loss + args.alpha * ctr_loss + im_div_loss
            loss = clf_loss + args.alpha * ctr_loss 
            one_epoch_loss += loss.item() # 一个epoch的loss累加
            loss.backward()
            # multiple (1./alpha) in order to remove the effect of alpha on updating centers
            for param in ctr_loss_func.parameters():
                try:
                    param.grad.data *= (1./args.alpha) 
                except:
                    pass
            optimizer.step()
            # scheduler.step()
            pred_cls = torch.argmax(output, dim=1)  # 置信度最高的类别
            train_correct += float((pred_cls ==b_y).sum())
        one_epoch_loss_avg = one_epoch_loss / len(trainloader) # batch数
        loss_item_train.append(one_epoch_loss_avg)
        train_acc = train_correct / len(trainloader.dataset) # trial数
        train_acc_all.append(train_acc)

        # log loss and acc
        # wandb.log({"train_acc":train_acc, "train_loss":one_epoch_loss_avg})

        # 每个epoch的验证集
        with torch.no_grad():
            val_loss, val_acc, val_f1, val_eer = evaluate_centerloss(model,valloader,args)
            # wandb.log({"val_acc":val_acc, "val_loss":val_loss})
        # 注释掉这两行,则不早停
        if (epoch - best_epoch) > args.earlystop: # 早停 
            break
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 
            os.path.join(savepath,f"{args.model}_{args.seed}_ctrloss_{args.alpha}.pth"))
        val_acc_all.append(val_acc)  # 验证集准确率list
        loss_item_val.append(val_loss)

        # 打印训练损失
        if (epoch + 1) % 10 == 0 :
            print(f"Epoch:{epoch+1}\tTrain_acc:{train_acc:.4f}\tVal_acc:{val_acc:.4f}\tTrain_loss:{one_epoch_loss_avg:.6f}\tVal_loss:{val_loss:.6f}")

    print( "-" * 20 + "训练完成!" + "-" * 20)
    print(f"总训练轮数-{epoch+1}, 早停轮数-{best_epoch+1}")
    print(f"验证集最佳准确率为{best_acc*100:.2f}%")
    print(f"验证集平均准确率为{np.mean(np.array(val_acc_all))*100:.2f}%")
    return model

def main():
    parser = argparse.ArgumentParser(description="Model Train Hyperparameter")
    parser.add_argument("--root", type=str, default="/mnt/data1/tyl/UserID/dataset/mydata")
    # "001","004","BCI85","Rest85", "LJ30", OpenBMI:"Rest", "MI", "ERP", "SSVEP", "cross"
    parser.add_argument("--setsplit", type=str, default=None) 
    parser.add_argument("--gpuid", type=int, default=9)
    parser.add_argument("--nclass", type=int, default=9) # 用户数量
    parser.add_argument("--channel", type=int, default=22)
    parser.add_argument("--timepoint", type=int, default=4)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--initlr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--earlystop", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--model", type=str, default="ResEEGNet") 
    # EEGNet,ShallowConvNet,DeepConvNet,ResEEGNet,FBCNet,Conformer
    parser.add_argument("--alpha", type=float, default=1e-3) # 损失加权项
    parser.add_argument("--aug_type", type=str, default=None) # 数据增强类型
    parser.add_argument("--resblock", type=int, default=5) # 数据增强类型
    # shard, channel_mixure, channel_mixup, trial_mixup, channel_reverse, channel_noise
    parser.add_argument("--ctr_loss", type=str2bool, nargs='?', default=False, help="是否用centerloss训练") # 是否用centerloss训练
    parser.add_argument("--use_EA", type=bool, default=False, help="是否用EA处理(GPU上)") # 是否用centerloss训练
    parser.add_argument("--strideFactor", type=int, default=4, help="尺度因子数") # 是否用centerloss训练
    parser.add_argument("--session_num", type=int, default=1, help="跨任务中使用第几个session数据") # 是否用centerloss训练
    parser.add_argument("--cross_task", type=str, nargs='+', default=None, help="跨任务") # 
    parser.add_argument("--tl", type=str, default=None, help="迁移学习方法") # 

    args = parser.parse_args()

    set_args(args)

    results = np.zeros((5,3)) # acc, f1, eer

    sys.stdout = Logger(os.path.join("/mnt/data1/tyl/UserID/logdir", 
                                     f"{args.setsplit}_" +  f'WithinSession_{args.model}.log'))

    for idx,seed in enumerate(range(2024, 2029)): 
        args.seed = seed
        start_time = time.time()
        print("="*30)
        print(f"dataset: {args.setsplit}")
        print(f"model  : {args.model}")
        print(f"seed   : {args.seed}")
        print(f"gpu    : {args.gpuid}")
        # print(f"aug    : {args.aug_type}")
        # print(f"attack : {args.attack_type}")
        set_seed(args.seed)
        # start a new experiment
        # wandb.init(project=f"{args.model}-{args.setsplit}")
        # wandb.config = {"learning_rate": args.initlr,
        #                 "epochs": args.epoch, 
        #                 "batch_size": args.bs, 
        #                 "early_stopping": args.earlystop, 
        #                 "seed": args.seed}
        trainloader, valloader, testloader = GetLoaderWithinSessionM3CV(seed=args.seed, split=args.setsplit, )

        print("=====================data are prepared===============")
        print(f"累计用时{time.time()-start_time:.4f}s!")

        # where to save the model
        model_path = os.path.join("/mnt/data1/tyl/UserID/ModelSave", f"{args.setsplit}")
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if not args.ctr_loss:
            model = train(trainloader,valloader,model_path,args)
            print("=====================model are trained===============")
            print(f"累计用时{time.time()-start_time:.4f}s!")
            test_loss, test_acc, test_f1, test_eer = evaluate(model,testloader,args)
        else:
            model = train_centerloss(trainloader,valloader,model_path,args)
            print("=====================model are trained===============")
            print(f"累计用时{time.time()-start_time:.4f}s!")
            test_loss, test_acc, test_f1, test_eer = evaluate_centerloss(model,testloader,args)

        # 统计模型参数
        # summary(model, input_size=(args.bs,args.channel,args.fs*args.timepoint))
        # end log
        # wandb.finish()
        results[idx] = [test_acc, test_f1, test_eer]
        print(f"测试集平均指标为  Acc:{test_acc*100:.2f}%;  F1:{test_f1*100:.2f}%;  EER:{test_eer*100:.2f}%;")
        print("=====================test are done===================")

        row_labels = ['2024', '2025', '2026', '2027', '2028', "Avg", "Std"]
        col_labels = ['Acc', 'F1', 'EER']
        print(f"训练集:验证集:测试集={len(trainloader.dataset)}:{len(valloader.dataset)}:{len(testloader.dataset)}")
        # 打印列标签
        print(f"{'SEED':<10} {col_labels[0]:<10} {col_labels[1]:<10} {col_labels[2]:<10}")
        # 打印每一行数据，包括行标签
        for i, row in enumerate(results):
            print(f"{row_labels[i]:<10} {row[0]:<10.4f} {row[1]:<10.4f} {row[2]:<10.4f}")
        print(f"{row_labels[-2]:<10} {np.mean(results[:idx+1,0]):<10.4f} {np.mean(results[:idx+1,1]):<10.4f} {np.mean(results[:idx+1,2]):<10.4f}")
        print(f"{row_labels[-1]:<10} {np.std(results[:idx+1,0]):<10.4f} {np.std(results[:idx+1,1]):<10.4f} {np.std(results[:idx+1,2]):<10.4f}")
        gc.collect()

    print("-"*50)
    print(model)

    final_results = np.vstack([results, np.mean(results, axis=0), np.std(results, axis=0)])
    df = pd.DataFrame(final_results, 
                  columns=['Acc', 'F1', 'EER'],
                  index=['2024', '2025', '2026', '2027', '2028', "Avg", "Std"])
    df = df.round(4)
    csv_path = os.path.join("/mnt/data1/tyl/UserID/csv", f"{args.setsplit}")
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    df.to_csv(os.path.join(csv_path, f"WithinSession_{args.model}.csv"))

if __name__ == "__main__":
    main()


