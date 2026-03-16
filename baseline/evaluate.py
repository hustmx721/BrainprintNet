import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("/mnt/data1/tyl/UserID/")

import numpy as np
from utils.loss import CenterLoss
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize, OneHotEncoder
from tlutils.loss import CELabelSmooth
from dataloader import SubBandSplit


def calculate_eer(y_true, y_pred):
    """
    计算 Equal Error Rate (EER)
    
    参数:
    y_true: 真实标签 (整数编码)
    y_pred: 预测的类别 (整数编码)
    
    返回:
    avg_eer: 平均 EER
    class_eer: 每个类别的 EER
    """
    # 获取唯一的类别
    classes = np.unique(np.concatenate((y_true)))
    n_classes = len(classes)
    
    # 将真实标签和预测标签转换为 one-hot 编码
    encoder = OneHotEncoder(sparse_output=False)
    y_true_bin = encoder.fit_transform(y_true.reshape(-1, 1))
    y_pred_bin = encoder.transform(y_pred.reshape(-1, 1))
    
    class_eer = []
    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = np.mean([fpr[np.nanargmin(np.absolute((fnr - fpr)))], 
                       fnr[np.nanargmin(np.absolute((fnr - fpr)))]])
        class_eer.append(eer)
    
    avg_eer = np.mean(class_eer)
    return avg_eer, class_eer


def evaluate(model, dataloader, args):
    model.eval()
    correct, val_loss= 0, 0
    total = len(dataloader.dataset) # loader中的总trial数
    device = torch.device("cuda:"+str(args.gpuid) if torch.cuda.is_available() else "cpu")
    clf_loss_func = nn.CrossEntropyLoss().to(device)
    # clf_loss_func = CELabelSmooth(num_classes=args.nclass).to(device)
    prob, y_pred, y_true = [], [], []
    for x, y in dataloader: # data，label
        x = x.cuda()
        y = y.cuda()

        with torch.no_grad():
            features, logits = model(x) # 预测结果，返回每个类别预测概率
            pred_y = torch.max(logits, 1)[1] 
            # loss = clf_loss_func(logits, F.one_hot(y.long(), args.nclass)) # CELabelSmooth
            loss = clf_loss_func(logits, y.long())
            correct += float((pred_y == y).sum())
            val_loss += loss
            prob.extend(logits.softmax(1).detach().cpu().tolist())
            y_pred.extend(pred_y.detach().long().cpu().tolist())
            y_true.extend(y.detach().long().cpu().tolist())
    
    # cm = confusion_matrix(y_true, y_pred)
    val_loss = val_loss / len(dataloader) # 取平均损失
    val_acc = correct / total
    val_f1 = f1_score(np.array(y_true), np.array(y_pred), average='weighted')
    val_eer ,_ = calculate_eer(np.expand_dims(np.array(y_true), axis=0), np.expand_dims(np.array(y_pred), axis=0))

    return val_loss, val_acc, val_f1, val_eer

    
def evaluate_centerloss(model, dataloader, args):
    model.eval()
    correct, val_loss= 0, 0
    total = len(dataloader.dataset) # loader中的总trial数s
    device = torch.device("cuda:"+str(args.gpuid) if torch.cuda.is_available() else "cpu")
    fea_dim = model._get_fea_dim(torch.randn(args.bs,1,args.channel,int(args.fs*args.timepoint)).to(device))[-1]
    clf_loss_func = nn.CrossEntropyLoss().to(device)
    ctr_loss_func = CenterLoss(num_classes=args.nclass,feat_dim=fea_dim,use_gpu=True)
    y_pred, y_true = [], []

    for x, y in dataloader: # data，label
        x = x.cuda()
        y = y.cuda()

        with torch.no_grad():
            features, logits = model.forward(x) # 预测结果，返回每个类别预测概率
            pred_y = torch.max(logits, 1)[1] 
            # torch.max()函数返回值为元组,(tensor,index),这里返回索引(0 1 2)
            # 对于损失而言，会转化为one-hot编码，则加不加一无所谓，[0,1,0](这种)
            clf_loss = clf_loss_func(logits, y.long())
            ctr_loss = ctr_loss_func(features, y.long())
            loss = clf_loss 
            loss = clf_loss + args.alpha * ctr_loss
    
            correct += float((pred_y == y).sum())
            val_loss += loss
            y_pred.extend(pred_y.detach().long().cpu().tolist())
            y_true.extend(y.detach().long().cpu().tolist())

    # [y_pred,y_true] = [np.array(y).reshape(-1) for y in [y_pred,y_true]]
    # conf_mat = confusion_matrix(y_true,y_pred,labels=np.unique(y_true))
    # draw confusion matrix
    """ 
    fig, ax = plt.subplots(figsize=(10,7))
    sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt="d")
    ax.set_title('confusion matrix')
    plt.xlabel("Predicted")
    plt.ylabel("Reference")
    plt.savefig(f'/data2/tyl/baselines/UserId/visual/{args.model}_{args.setsplit}_confmat.png', bbox_inches='tight',dpi=300,pad_inches=0)
    plt.show() 
    """

    val_loss = val_loss / len(dataloader) # 取平均损失
    val_acc = correct / total
    val_f1 = f1_score(np.array(y_true), np.array(y_pred), average='weighted')
    val_eer ,_ = calculate_eer(np.expand_dims(np.array(y_true), axis=0), np.expand_dims(np.array(y_pred), axis=0))

    return val_loss, val_acc, val_f1, val_eer   
