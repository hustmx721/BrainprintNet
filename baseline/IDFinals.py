"""
决赛最终版本, ID分类训练代码 
"""

import argparse
import os
import torch
import sys
import time
sys.path.append("/data2/tyl/baselines/UserId")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader

from utils.utlis import str2bool
from utils.data_augment import shard, channel_mixup, trial_mixup, channel_reverse, channel_noise
from utils.preprocess import preprocessing
from utils.mylogging import Logger
from utils.Attack import AWP, FGSM
from baseline.dataloader import process_label, raw2dataFinals
from dataset.instance import set_seed
from models.constraintmodels import MyConformerWithConstraint
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split




def evaluate(model, dataloader, if_test, args):
    model.eval()
    correct, val_loss= 0, 0
    total = len(dataloader.dataset) # loader中的总trial数
    device = torch.device("cuda:"+str(args.gpuid) if torch.cuda.is_available() else "cpu")
    clf_loss_func = nn.CrossEntropyLoss().to(device)
    prob, y_pred, y_true, s_true = [], [], [], []
    for x, y, s in dataloader: # data，label
        x = x.cuda()
        y = y.cuda()

        with torch.no_grad():
            features, logits = model(x) # 预测结果，返回每个类别预测概率
            pred_y = torch.max(logits, 1)[1] 
            loss = clf_loss_func(logits, y.long())
            correct += float((pred_y == y).sum())
            val_loss += loss
            prob.extend(logits.softmax(1).detach().cpu().tolist())
            y_pred.extend(pred_y.detach().long().cpu().tolist())
            y_true.extend(y.detach().long().cpu().tolist())
            s_true.extend(s.tolist())

    val_loss = val_loss / len(dataloader) # 取平均损失
    val_acc = correct / total
    val_f1 = f1_score(np.array(y_true), np.array(y_pred), average='weighted')
    val_bca = balanced_accuracy_score(np.array(y_true), np.array(y_pred))
    
    # 验证模式
    if not if_test:
        return val_loss, val_acc, val_f1, val_bca
    # 测试模式
    elif if_test:
        result = zip(prob, y_true, y_pred, s_true)
        real_cnt, fake_cnt = 0, 0
        for i, (logit, y_t, y_p, s) in enumerate(result):
            if s == 1 and y_t == y_p:
                real_cnt += 1
            elif s == 0 and y_t == y_p:
                fake_cnt += 1
        s_true = np.array(s_true)
        real_acc, fake_acc = real_cnt/(len(s_true[np.where(s_true==1)])+1e-10), fake_cnt/(len(s_true[np.where(s_true==0)])+1e-10)
        print(f"real_acc: {real_acc*100:.2f}%")
        print(f"fake_acc: {fake_acc*100:.2f}%")
        return val_loss, val_acc, val_f1, val_bca, real_acc, fake_acc

# 生成数据集   
class MyDataset(Dataset):
    def __init__(self,data,label,rf):
        self.data = data
        self.label = label
        self.rf = rf    
    
    def __getitem__(self, idx):
        data = self.data[idx].astype(np.float32)
        label = self.label[idx]
        rf = self.rf[idx]
        return data, label, rf
    
    def __len__(self):
        return self.data.shape[0]


import numpy as np
from typing import Tuple, List

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


# 生成dataloader
def trans_torch(data,label,rf,mode): # "train"/"test"
    data_set = MyDataset(data,label,rf)
    if mode == "train":
        dataloader = DataLoader(dataset = data_set,batch_size = 512, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    elif mode == "test":
        dataloader = DataLoader(dataset = data_set,batch_size = 128, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    return dataloader



def ID_final_loader(seed,len):
    # 模拟线上,先切成3s没有重叠的数据
    data, label = raw2dataFinals(fs=250,len=3)
    label = process_label(label) - 1
    for idx,y in enumerate(label):
        if y == 1 or y == 15 :
            aug_cls2_x, aug_cls2_y = channel_noise(np.expand_dims(data[idx],axis=0),label[idx],multiplier=9,noise_type="pink")
            data, label = np.vstack((data,aug_cls2_x)), np.hstack((label,aug_cls2_y))
        if y == 8 or y == 19 :  
            aug_cls16_x, aug_cls16_y = channel_noise(np.expand_dims(data[idx],axis=0),label[idx],multiplier=2,noise_type="pink")
            data, label = np.vstack((data,aug_cls16_x)), np.hstack((label,aug_cls16_y))
        if y == 5 or y == 9 or y == 13 :  
            aug_cls16_x, aug_cls16_y = channel_noise(np.expand_dims(data[idx],axis=0),label[idx],multiplier=1,noise_type="pink")
            data, label = np.vstack((data,aug_cls16_x)), np.hstack((label,aug_cls16_y))
        else:
            continue
    rf = np.ones(label.shape[0])
    tx, ex, ty, ey, ts, es = train_test_split(data, label, rf, test_size=0.2, random_state=seed, stratify=label)
    # 滑窗数据增强
    tx, ty = shard(data=tx,label=ty,postdata_len=int(len*250),strip=50,if_aug=True)
    ex, ey = shard(data=ex,label=ey,postdata_len=int(len*250),strip=50,if_aug=True)
    ts, es = np.ones(tx.shape[0]), np.ones(ex.shape[0])
    vx, ex, vy, ey, vs, es = train_test_split(ex, ey, es, test_size=0.5, random_state=seed, stratify=ey)
    # 对训练数据进行数据增强
    splits = stratified_split(X=tx, y=ty, splits=[0.25]*4, random_state=seed)
    Method = [channel_mixup, trial_mixup, channel_reverse, channel_noise]
    aug_x, aug_y = [], []
    for data, method in zip(splits, Method):
        x, y  = data[0], data[1]
        x, y= method(data=x,labels=y,multiplier=1)
        aug_x.append(x)
        aug_y.append(y)
    tx, ty = np.concatenate((tx, *aug_x),axis=0), np.concatenate((ty, *aug_y),axis=0)
    ts = np.ones(tx.shape[0])
    DataProcessor = preprocessing(fs=250) # 001/004采样率
    [tx, vx, ex] = [DataProcessor.Bandpassfilter(x,4,40) for x in [tx, vx, ex]]
    [tx, vx, ex] = [np.expand_dims(x,axis=1) for x in [tx, vx, ex]]

    print("训练集:验证集:测试集--", tx.shape,vx.shape,ex.shape)
    [trainloader, validateloader, testloader] = [trans_torch(x,y,s,mode) \
     for x,y,s,mode in zip([tx, vx, ex], [ty, vy, ey], [ts, vs, es], ["train","test", "test"])]
    return trainloader, validateloader, testloader


def train(trainloader,valloader,savepath,args):
    print( "-" * 20 + "开始训练!" + "-" * 20)
    device = torch.device("cuda:"+str(args.gpuid) if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.cuda.set_device(device)
    model = MyConformerWithConstraint(kernels=[11,21,31,41,51],n_classes=args.nclass,
                        Chans=args.channel,Samples=int(args.fs*args.timepoint),
                        emb_size=64,depth=4).to(device)
  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initlr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,140,180], gamma=0.5)
    clf_loss_func = nn.CrossEntropyLoss().to(device)
    # awp = AWP(model, optimizer)

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
        for _,(b_x,b_y,b_s) in enumerate(trainloader):
            model.train()
            optimizer.zero_grad()
            
            b_x, b_y = b_x.to(device), b_y.to(device)
            features, output = model(b_x) # 输出每个类别的概率

            loss = clf_loss_func(output,b_y.long())

            one_epoch_loss += loss.item() # 一个epoch的loss累加
            loss.backward()


            optimizer.step()

            pred_cls = torch.argmax(output, dim=1)  # 置信度最高的类别
            train_correct += float((pred_cls == b_y).sum())
        one_epoch_loss_avg = one_epoch_loss / len(trainloader) # batch数
        loss_item_train.append(one_epoch_loss_avg)
        train_acc = train_correct / len(trainloader.dataset) # trial数
        train_acc_all.append(train_acc)

        # 每个epoch的验证集
        with torch.no_grad():
            val_loss, val_acc, _, _ = evaluate(model=model,dataloader=valloader,if_test=False,args=args)
        # 注释掉这两行,则不早停
        if (epoch - best_epoch) > args.earlystop: # 早停 
            break
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(savepath,f"{args.setsplit}_Finals_{args.timepoint}s_{args.model}_{args.seed}.pth"))
        val_acc_all.append(val_acc)  # 验证集准确率list
        loss_item_val.append(val_loss)
        # scheduler.step()

        # 打印训练损失
        if (epoch + 1) % 10 == 0 :
            print(f"Epoch:{epoch+1}\tTrain_acc:{train_acc:.4f}\tVal_acc:{val_acc:.4f}\tTrain_loss:{one_epoch_loss_avg:.6f}\tVal_loss:{val_loss:.6f}")

    print( "-" * 20 + "训练完成!" + "-" * 20)
    print(f"总训练轮数-{epoch+1}, 早停轮数-{best_epoch+1}")
    print(f"验证集最佳准确率为{best_acc*100:.2f}%")
    print(f"验证集平均准确率为{np.mean(np.array(val_acc_all))*100:.2f}%")
    return model


def main_dl():
    parser = argparse.ArgumentParser(description="Model Train Hyperparameter")
    parser.add_argument("--setsplit", type=str, default="id") # "rf", "id"
    parser.add_argument("--nclass", type=int, default=23) # 用户数量
    parser.add_argument("--gpuid", type=int, default=2)
    parser.add_argument("--timepoint", type=float, default=3) # 使用数据长度
    parser.add_argument("--channel", type=int, default=5)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--initlr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--earlystop", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--model", type=str, default="MyConformerWithConstraint") 
    parser.add_argument("--hidsize", type=int, default=50) # 分类器中隐藏层神经元数量
    parser.add_argument("--alpha", type=float, default=0.001) # 是否重训练rf模型
    parser.add_argument("--aug_type", type=str, default="channel_noise") # 数据增强类型
    # shard, channel_mixure, channel_mixup, trial_mixup, channel_reverse, channel_noise
    parser.add_argument("--ctl_loss", type=str2bool, nargs='?', default=True, help="是否用centerloss训练") # 是否用centerloss训练

    args = parser.parse_args()

    result = np.zeros((7,5))
 
    sys.stdout = Logger(os.path.join("/data2/tyl/competition/BCIC2024/UID/data/src/logdir", 
                                        f"{args.setsplit}_Finals_{args.timepoint}s_{args.model}.log"))
    
    for idx,seed in enumerate(range(2024,2031)):
        args.seed = seed
        start_time = time.time()
        print("="*30)
        print(f"task   : {args.setsplit}")
        print(f"model  : {args.model}")
        print(f"seed   : {args.seed}")
        print(f"gpu    : {args.gpuid}")
        print(f"time   : {args.timepoint}s")
        set_seed(args.seed)

        trainloader, valloader, testloader = ID_final_loader(args.seed,len=args.timepoint)

        print("=====================data are prepared===============")
        print(f"累计用时{time.time()-start_time:.4f}s!")

        # where to save the model
        model_path = os.path.join("/data2/tyl/competition/BCIC2024/UID/data/src/ModelSave")
        if not os.path.exists(model_path):
            os.makedirs(model_path)


        model = train(trainloader,valloader,model_path,args)
        _, test_acc, test_f1, test_bca, real_acc, fake_acc = evaluate(model=model,dataloader=testloader,if_test=True, args=args)


        print("=====================model are trained===============")
        print(f"累计用时{time.time()-start_time:.4f}s!")

        result[idx,:] = [test_acc,test_bca,test_f1,real_acc,fake_acc]
        print(f"测试集平均指标为  ACC:{test_acc*100:.2f}%;  BCA:{test_bca*100:.2f}%;  F1:{test_f1*100:.2f}%;  真实ID识别率:{real_acc*100:.2f}%;  合成ID识别率:{fake_acc*100:.2f}%")
        print("=====================test are done===================")

        row_labels = ['2024', '2025', '2026', '2027', '2028', '2029', '2030']
        col_labels = ['acc', 'bca', 'f1', "real_acc", "fake_acc"]
        print(f"{args.timepoint}s模型")
        print(f"训练集:验证集:测试集={len(trainloader.dataset)}:{len(valloader.dataset)}:{len(testloader.dataset)}")
        # 打印列标签
        print(f"{col_labels[0]:<15} {col_labels[1]:<15} {col_labels[2]:<15} {col_labels[3]:<15} {col_labels[4]:<15}")
        # 打印每一行数据，包括行标签
        for i, row in enumerate(result):
            print(f"{row_labels[i]:<10} {row[0]:<10.4f} {row[1]:<10.4f} {row[2]:<10.4f} {row[3]:<10.4f} {row[4]:<10.4f}")
   

if __name__ == "__main__":
    main_dl()
    # trainloader, valloader, testloader = ID_final_loader(2024,len=2)


