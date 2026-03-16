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
import pandas as pd
from tqdm import tqdm

from baseline.dataloader import *
from dataset.instance import set_seed
from evaluate import evaluate
from utils.utlis import load_baseline_model, load_data, set_args, str2bool
from utils.mylogging import Logger
from tlutils.network import feat_classifier, AdversarialNetwork, calc_coeff
from tlutils.loss import CELabelSmooth_raw, Entropy, ReverseLayerF, CDANE, RandomLayer


def train(trainloader,valloader, testloader, savepath,args):
    print( "-" * 20 + "开始训练!" + "-" * 20)
    device, models, optimizer = load_baseline_model(args)
    torch.cuda.empty_cache()
    torch.cuda.set_device(device)
    # loss funcs
    netF, netC = models
    fea_dim = netF._get_fea_dim(torch.randn(2, args.channel, int(args.fs*args.timepoint)).to(device))[-1]
    # TODO: 增加args.use_random_layer 超参数做判断条件
    if False:
        ad_net = AdversarialNetwork(fea_dim, 32, 8)
        random_layer = RandomLayer([fea_dim, args.nclass], fea_dim,
                                   use_cuda=True)
    else:
        ad_net = AdversarialNetwork(fea_dim * args.nclass, 32, 8)
        random_layer = None
        
    ad_net.to(device)
    netF.to(device)
    netC.to(device)
    model = nn.Sequential(netF, netC).to(device)

    optimizer_f = torch.optim.Adam(netF.parameters(), lr=args.initlr)
    optimizer_c = torch.optim.Adam(netC.parameters(), lr=args.initlr)
    optimizer_d = torch.optim.Adam(ad_net.parameters(), lr=args.initlr)

    clf_loss_func = nn.CrossEntropyLoss().to(device)
    
    # 训练指标
    best_epoch = 0
    best_acc = 0 # in validate dataset
    train_acc_all = []
    val_acc_all = []
    loss_item_train = []
    loss_item_val = []

    max_iter = len(trainloader) # 每个epoch的最大迭代次数

    for epoch in tqdm(range(args.epoch),desc="Training:"):
        model.train()
        train_correct = 0
        one_epoch_loss = 0
        iter_num = 0

        while iter_num < max_iter:
            try:
                x_src, y_src = next(iter_src)
            except:
                iter_src = iter(trainloader)
                x_src, y_src = next(iter_src)
            try:
                x_tar, _ = next(iter_tar)
            except:
                iter_tar = iter(testloader)
                x_tar, _ = next(iter_tar)

            x_src, y_src = x_src.to(device), y_src.to(device)
            x_tar = x_tar.to(device)

            iter_num += 1

            fea_src, logits_src = model(x_src)
            fea_tar, logits_tar = model(x_tar)
            fea = torch.cat((fea_src, fea_tar), dim=0)
            out = torch.cat((logits_src, logits_tar), dim=0)
            softmax_out = nn.Softmax(dim=1)(out)
            entropy = Entropy(softmax_out)
            transfer_loss = CDANE([fea, softmax_out], ad_net, entropy, calc_coeff(iter_num), args, random_layer)
            loss_trade_off = 1.0

            clf_loss = clf_loss_func(logits_src, y_src.long())
            
            total_loss = clf_loss + loss_trade_off * transfer_loss

            optimizer_f.zero_grad()
            optimizer_c.zero_grad()
            optimizer_d.zero_grad()
            total_loss.backward()
            optimizer_f.step()
            optimizer_c.step()
            optimizer_d.step()

            pred_cls = torch.argmax(logits_src, dim=1)
            one_epoch_loss += total_loss.item()
            train_correct += float((pred_cls == y_src).sum())

        one_epoch_loss_avg = one_epoch_loss / max_iter
        loss_item_train.append(one_epoch_loss_avg)
        train_acc = train_correct / len(trainloader.dataset)
        train_acc_all.append(train_acc)

        with torch.no_grad():
            model.eval()
            val_loss, val_acc, val_f1, val_eer = evaluate(model,valloader,args)

        if (epoch - best_epoch) > args.earlystop: # 早停 
            break
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 
            os.path.join(savepath, f"base_{args.tl}_{args.model}_session{args.session_num}_{args.seed}.pth"))
        val_acc_all.append(val_acc)  # 验证集准确率list
        loss_item_val.append(val_loss)

        # 打印训练损失
        if (epoch + 1) % 10 == 0 :
            print(f"Epoch:{epoch+1}\tTrain_acc:{train_acc:.4f}\tVal_acc:{val_acc:.4f}\tTrain_loss:{one_epoch_loss_avg:.6f}\tVal_loss:{val_loss:.6f}")

    print( "-" * 20 + "训练完成!" + "-" * 20)
    print(f"总训练轮数-{epoch+1}, 早停轮数-{best_epoch+1}")
    print(f"验证集最佳准确率为{best_acc*100:.2f}%")
    print(f"验证集平均准确率为{np.mean(np.array(val_acc_all))*100:.2f}%")
    gc.collect()
    torch.cuda.empty_cache()
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
    parser.add_argument("--aug_type", type=str, default="channel_noise") # 数据增强类型
    parser.add_argument("--resblock", type=int, default=5) # 数据增强类型
    # shard, channel_mixure, channel_mixup, trial_mixup, channel_reverse, channel_noise
    parser.add_argument("--ctr_loss", type=str2bool, nargs='?', default=False, help="是否用centerloss训练") # 是否用centerloss训练
    parser.add_argument("--use_EA", type=bool, default=False, help="是否用EA处理(GPU上)") # 
    parser.add_argument("--strideFactor", type=int, default=4, help="尺度因子数") # 
    parser.add_argument("--session_num", type=int, default=1, help="跨任务中使用第几个session数据") # 
    parser.add_argument("--cross_task", type=str, nargs='+', default=None, help="跨任务") # 
    parser.add_argument("--tl", type=str, default=None, help="迁移学习方法") # 
    parser.add_argument("--alignment_weight", type=float, default=1.0, help="对齐损失权重") # 
    args = parser.parse_args()

    set_args(args)

    results = np.zeros((5,3)) # acc, f1, eer

    if args.cross_task is not None and args.cross_task[0].startswith("M3CV"):
        cross_task = args.cross_task[0][5:] + "_" + args.cross_task[1][5:]
    elif args.cross_task is not None and args.cross_task[0] in ["MI", "ERP", "SSVEP"]:
        cross_task = args.cross_task[0] + "_" + args.cross_task[1]

    sys.stdout = Logger(os.path.join("/mnt/data1/tyl/UserID/logdir", 
                                     f"{cross_task}_{args.tl}_" +  f'{args.model}_base.log'))

    for idx,seed in enumerate(range(2024, 2029)): 
        args.seed = seed
        start_time = time.time()
        print("="*30)
        print(f"dataset: {cross_task}")
        print(f"model  : {args.model}")
        print(f"seed   : {args.seed}")
        print(f"gpu    : {args.gpuid}")
        set_seed(args.seed)

        trainloader, valloader, testloader = load_data(args)

        print("=====================data are prepared===============")
        print(f"累计用时{time.time()-start_time:.4f}s!")

        model_path = os.path.join("/mnt/data1/tyl/UserID/ModelSave", f"{cross_task}")
        if not os.path.exists(model_path):
            os.makedirs(model_path)


        model = train(trainloader,valloader,testloader, model_path,args)
        print("=====================model are trained===============")
        print(f"累计用时{time.time()-start_time:.4f}s!")
        model.eval()
        test_loss, test_acc, test_f1, test_eer = evaluate(model,testloader,args)

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
    csv_path = os.path.join("/mnt/data1/tyl/UserID/csv", f"{cross_task}")
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    df.to_csv(os.path.join(csv_path, f"base_{args.tl}_{args.model}_session{args.session_num}.csv"))

if __name__ == "__main__":
    main()


