import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import time
import wandb
import gc
sys.path.append("/mnt/data1/tyl/UserID/")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist

from baseline.dataloader import *
from dataset.instance import set_seed
from evaluate import evaluate
from utils.utlis import load_baseline_model, load_data, set_args, str2bool
from utils.mylogging import Logger
from tlutils.loss import ClassConfusionLoss, Entropy

def obtain_label(loader, netF, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netF(inputs)
            feas, outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
    unknown_weight = 1 - ent / np.log(args.nclass)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # TODO: 默认使用cosine距离
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>0) # TODO: args.threshold默认设置为0
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], "cosine")
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    print(log_str+'\n')

    return predict.astype('int')


def train_srcmodel(trainloader, valloader, testloader, savepath,args):
    print( "-" * 20 + "开始训练!" + "-" * 20)
    device, models, optimizer = load_baseline_model(args)
    torch.cuda.empty_cache()
    torch.cuda.set_device(device)
    # loss funcs
    netF, netC = models

    netF.to(device)
    netC.to(device)
    model = nn.Sequential(netF, netC).to(device)

    optimizer_f = torch.optim.Adam(netF.parameters(), lr=args.initlr)
    optimizer_c = torch.optim.Adam(netC.parameters(), lr=args.initlr)

    clf_loss_func = nn.CrossEntropyLoss().to(device)
    
    ######################################################################################################
    # Source Model Training
    ######################################################################################################

    # 训练指标
    best_epoch = 0
    best_acc = 0 # in validate dataset
    train_acc_all = []
    val_acc_all = []
    loss_item_train = []
    loss_item_val = []

    max_iter = len(trainloader) # 每个epoch的最大迭代次数

    # TODO: load pretrained model
    print(" Source Model Training ")
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

            x_src, y_src = x_src.to(device), y_src.to(device)

            iter_num += 1

            fea_src, logits_src = model(x_src)
            clf_loss = clf_loss_func(logits_src, y_src.long())
            total_loss = clf_loss

            optimizer_f.zero_grad()
            optimizer_c.zero_grad()
            total_loss.backward()
            optimizer_f.step()
            optimizer_c.step()

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
        val_acc_all.append(val_acc)  # 验证集准确率list
        loss_item_val.append(val_loss)

        # 打印训练损失
        if (epoch + 1) % 10 == 0 :
            print(f"Epoch:{epoch+1}\tTrain_acc:{train_acc:.4f}\tVal_acc:{val_acc:.4f}\tTrain_loss:{one_epoch_loss_avg:.6f}\tVal_loss:{val_loss:.6f}")

    print( "-" * 20 + "训练完成!" + "-" * 20)
    print(f"总训练轮数-{epoch+1}, 早停轮数-{best_epoch+1}")
    print(f"验证集最佳准确率为{best_acc*100:.2f}%")
    print(f"验证集平均准确率为{np.mean(np.array(val_acc_all))*100:.2f}%")

    ######################################################################################################
    # Source HypOthesis Transfer
    ######################################################################################################

    print('Source HypOthesis Transfer')
    netF.eval()
    netC.eval()
    # SHOT重新目标域训练分类器最大的epoch数
    cls_par = 0
    max_epoch = 5
    batch_size = 64 # 默认batchsize大小
    interval = 5 # 每隔interval个epoch进行一次测试
    ent = False # 是否使用IM loss作为无监督损失
    gent = True
    ent_par = 1.0
    optimizer = optim.Adam(netF.parameters(), lr=args.initlr)

    max_iter = max_epoch * len(testloader)
    interval_iter = max_iter // interval
    iter_num = 0

    while iter_num < max_iter:
        try:
            x_tar, _ = next(iter_tar)
            tar_id += 1
            tar_idx = np.arange(batch_size, dtype=int) + batch_size * tar_id
        except:
            iter_tar = iter(testloader)
            x_tar, _ = next(iter_tar)
            tar_id = 0
            tar_idx = np.arange(batch_size, dtype=int)

        x_tar = x_tar.to(device)

        if iter_num % interval_iter == 0 and cls_par > 0:
            netF.eval()
            mem_label = obtain_label(testloader, netF, netC, args)
            mem_label = torch.from_numpy(mem_label).to(device)
            netF.train()
        
        iter_num += 1
        fea_tar = netF(x_tar)
        fea_tar, logits_tar = netC(fea_tar)

        # define loss
        if cls_par > 0:
            beta = 0.8
            py, y_prime = F.softmax(logits_tar, dim=-1).max(1)
            flag = py > beta
            clf_loss = F.cross_entropy(logits_tar[flag], y_prime[flag])

            clf_loss *= cls_par
        else:
            clf_loss = torch.tensor(0.0,requires_grad=True).to(device)
        
        if ent:
            softmax_out = nn.Softmax(dim=1)(logits_tar)
            entropy_loss = torch.mean(Entropy(softmax_out))
            if gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
                entropy_loss += gentropy_loss
            im_loss = entropy_loss * ent_par
            clf_loss = im_loss

        optimizer.zero_grad()
        clf_loss.backward()
        optimizer.step()
    
    model = nn.Sequential(netF, netC).to(device)
    torch.save(model.state_dict(), 
            os.path.join(savepath, f"base_{args.tl}_{args.model}_session{args.session_num}_{args.seed}.pth"))

    gc.collect()
    torch.cuda.empty_cache()

    print("Source HypOthesis Transfer Finished!")
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


        model = train_srcmodel(trainloader,valloader,testloader, model_path,args)
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


