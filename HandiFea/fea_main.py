""" 
instruction: use kind of feature extraction methods to predict user ID 
methods : 
    - WaveletPacket: 小波包分解特征
    - PSD: 功率谱密度和微分熵特征
    - AR_burg: 自回归移动平均系数
    - Entropy : 样本熵、近似熵、模糊熵
Author:hust-marx2
time: 2024/4/17
lastest:
"""

import sys
sys.path.append("/mnt/data1/tyl/UserID")

from utils.preprocess import preprocessing
from utils.mylogging import Logger
from handifea.fea import AR_burg, WaveletPacket, Entropy, PSD, trans_mfccs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_curve
from sklearn.preprocessing import OneHotEncoder


import os
import time
import argparse
import pickle
import numpy as np
import scipy.io as scio
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

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


def load_data(setsplit):
    start_time = time.time()
    t = 5 if setsplit in["BCI85","LJ30"] else 0.8 if setsplit == "ERP" else 4
    fs = 512 if setsplit == "BCI85" else 1000 if setsplit == "ERP" else 300 if setsplit == "LJ30" else 200 if setsplit == "SEED" else 250 
    
    if setsplit in ["001", "004"]:
        raw = scio.loadmat(f"/mnt/data1/tyl/UserID/dataset/mydata/ori_{setsplit}.mat")
    elif setsplit == "BCI85":
        raw = np.load("/mnt/data1/tyl/UserID/dataset/mydata/ori_BCI85.pkl",allow_pickle=True)
    elif setsplit == "LJ30":
        raw = scio.loadmat("/mnt/data1/tyl/UserID/dataset/mydata/ori_LingJiu30.mat")
    elif setsplit == "SEED":
        data1, data2 = pickle.load(open('/mnt/data1/tyl/data/SEED/processed/train.pkl',"rb")), \
        pickle.load(open('/mnt/data1/tyl/data/SEED/processed/test.pkl',"rb"))
        tx, ts = data1['data'].astype(np.float32), (data1['label']).astype(np.int16) -1
        ex, es = data2['data'].astype(np.float32), (data2['label']).astype(np.int16) -1
        print(f"数据加载完毕,累计用时{time.time()-start_time:.2f}s!")
        return tx, ts, ex, es, fs, t
    elif setsplit in ["MI", "ERP", "SSVEP"]:
        raw1 ,raw2 = pickle.load(open(f'/mnt/data1/tyl/data/OpenBMI/processed/{setsplit}/train.pkl',"rb")), \
        pickle.load(open(f'/mnt/data1/tyl/data/OpenBMI/processed/{setsplit}/test.pkl',"rb"))
        tx, ts = raw1["ori_train_x"].astype(np.float32), raw1["ori_train_s"]-1
        ex, es = raw2["ori_test_x"].astype(np.float32), raw2["ori_test_s"]-1
        tx, ex = [x.reshape((-1,x.shape[-2],x.shape[-1])) for x in [tx, ex]] # (trials, channels, samples)
        ts, es = [s.reshape(-1).astype(np.int16) for s in [ts, es]] # (trials,)
        print(f"数据加载完毕,累计用时{time.time()-start_time:.2f}s!")
        return tx, ts, ex, es, fs, t
    elif setsplit == "CatERP":
        raw1 ,raw2 = pickle.load(open(f'/mnt/data1/tyl/data/OpenBMI/processed/ERP/Cat_train.pkl',"rb")), \
        pickle.load(open(f'/mnt/data1/tyl/data/OpenBMI/processed/ERP/Cat_test.pkl',"rb"))
        tx, ts = raw1["ori_train_x"].astype(np.float32), raw1["ori_train_s"]-1
        ex, es = raw2["ori_test_x"].astype(np.float32), raw2["ori_test_s"]-1
        tx, ex = [x.reshape((-1,x.shape[-2],x.shape[-1])) for x in [tx, ex]] # (trials, channels, samples)
        ts, es = [s.reshape(-1).astype(np.int16) for s in [ts, es]] # (trials,)
        print(f"数据加载完毕,累计用时{time.time()-start_time:.2f}s!")
        return tx, ts, ex, es, fs, t
    elif setsplit in ["M3CV_Rest", "M3CV_Transient", "M3CV_Steady", "M3CV_P300", "M3CV_Motor", "M3CV_SSVEP_SA"]:
        split = setsplit[5:]
        with open(f'/mnt/data1/tyl/data/M3CV/Train/T_{split}.pkl', 'rb') as f1, open(f'/mnt/data1/tyl/data/M3CV/Test/{split}.pkl', 'rb') as f2:
            data1, data2 = pickle.load(f1), pickle.load(f2)
        # 去除EasyCap
        tx, ts = data1['data'][:,:-1,:].astype(np.float32), (data1['label']).astype(np.int16)
        ex, es = data2['data'][:,:-1,:].astype(np.float32), (data2['label']).astype(np.int16)
        print(f"数据加载完毕,累计用时{time.time()-start_time:.2f}s!")
        return tx, ts, ex, es, fs, t
    if setsplit in ["001", "004", "BCI85", "LJ30"]:
        tx, ex = raw["ori_train_x"], raw["ori_test_x"]
        ts, es = raw["ori_train_s"]-1, raw["ori_test_s"]-1
        tx, ex = [x.reshape((-1,x.shape[-2],x.shape[-1])) for x in [tx, ex]] # (trials, channels, samples)
        ts, es = [s.reshape(-1) for s in [ts, es]] # (trials,)

    print(f"数据加载完毕,累计用时{time.time()-start_time:.2f}s!")
   

    return tx, ts, ex, es, fs, t

# cross-paradigm data
def load_crossdata(split_train:str="MI", split_test:str="SSVEP"):
    start_time = time.time()
    t = 4
    fs = 250
    rawtrian = pickle.load(open(f'/mnt/data1/tyl/data/OpenBMI/processed/{split_train}/train.pkl',"rb"))
    rawtest = pickle.load(open(f'/mnt/data1/tyl/data/OpenBMI/processed/{split_test}/test.pkl',"rb"))
    tx, ts = rawtrian['ori_train_x'], rawtrian['ori_train_s']-1
    ex, es = rawtest['ori_test_x'], rawtest['ori_test_s']-1
    tx, ex = [x.reshape((-1,x.shape[-2],x.shape[-1])) for x in [tx, ex]]
    ts, es = [s.reshape(-1) for s in [ts, es]]
    processor = preprocessing(fs=250) # MI,SSVEP,Rest:250hz(4s)
    [tx, ex] = [processor.EEGpipline(x) for x in [tx, ex]]
    return tx, ts, ex, es, fs, t

    
def extract_features(tx, ex, ts, es, fs, t, fea_type):
    start_time = time.time()
    # shuffle data
    [shuffle_idx1, shuffle_idx2] = [np.random.permutation(x.shape[0]) for x in [tx,ex]]
    [tx, ts] = [x[shuffle_idx1] for x in [tx, ts]]
    [ex, es] = [x[shuffle_idx2] for x in [ex, es]]
    # preprocessing
    DataProcessor = preprocessing(fs=fs) 
    [tx, ex] = [DataProcessor.EEGpipline(x) for x in [tx, ex]] 
    # feature extraction
    if fea_type == "wavelet":
        [tf, ef] = [WaveletPacket(x) for x in [tx, ex]] 
    elif fea_type == "PSD":
        [[tf,_],[ef,_]] = \
        [PSD(x,{"stftn": 2*fs+100,"fStart": [1,4,8,14,31],"fStop": [3,7,13,30,50],"fs": fs, "EyeTime":t}) \
         for x in [tx, ex]]
    elif fea_type == "AR_burg":
        [tf, ef] = [AR_burg(x) for x in [tx, ex]]   
    elif fea_type == "Entropy":
        [tf, ef] = [Entropy(x) for x in [tx, ex]]
    elif fea_type == "MFCC":
        [tf, ef] = [trans_mfccs(wav_data=x,sample_rate=fs,framesize=256, mel_band=16, hop_length=128) for x in [tx, ex]]

    [tf, ef] = [fea.reshape((fea.shape[0],-1)).astype(np.float32) for fea in [tf, ef]]
    print(f"特征提取完毕,累计用时{time.time()-start_time:.2f}s!")

    tf[np.isnan(tf)], ef[np.isnan(ef)] = 0, 0
    tf[np.isinf(tf)], ef[np.isinf(ef)] = 1e6, 1e6

    return tf, ef, ts, es

def clf_predict(tf, ef, ts, es, clf_type):
    start_time = time.time()
    if clf_type == "LDA":
        clf = LDA()
    elif clf_type == "SVM":
        clf = svm.SVC()
    elif clf_type == "XGBoost":
        clf = XGBClassifier(objective="multi:softmax")

    clf.fit(tf, ts)
    print(f"分类器训练完毕,累计用时{time.time()-start_time:.2f}s!")
    pred = clf.predict(ef)
    acc = accuracy_score(es,pred)
    f1 = f1_score(es,pred,average="weighted")
    recall = recall_score(es,pred,average="weighted")
    precision = precision_score(es,pred,average="weighted")
    eer ,_ = calculate_eer(np.expand_dims(np.array(es), axis=0), np.expand_dims(np.array(pred), axis=0))
    

    return acc, f1, eer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Extraction and Classification Hyperparameter")
    # "001","004","BCI85","Rest85", "LingJiu30", OpenBMI:"Rest", "MI", "ERP", "SSVEP", "cross"
    parser.add_argument("--setsplit", type=str, default="ERP") 
    parser.add_argument("--clf_type", type=str, default="SVM") # "LDA", "SVM", "XGBoost"
    parser.add_argument("--fea_type", type=str, default="PSD") # "wavelet", "PSD", "AR_burg", "Entropy"
    parser.add_argument("--cross_type",type=str, nargs='+', default=None) #None:not cross-paradigm; List:["MI","SSVEP"] or ["SSVEP","MI"]
    # example: python -u fea_main.py  --fea_type=AR_burg --clf_type=SVM --cross_type MI SSVEP
    args = parser.parse_args()

    sys.stdout = Logger(os.path.join("/mnt/data1/tyl/UserID/logdir", 
                                     f'{args.setsplit}' + f"_{args.fea_type}" + f"_{args.clf_type}" \
                                     +'.log'))
    
    start_time = time.time()
    print("="*30)
    print(f"dataset: {args.setsplit}")
    print(f"feature: {args.fea_type}")
    print(f"classifier: {args.clf_type}")

    if args.cross_type is not None:
        print(f"cross-paradigm: train pdg--{args.cross_type[0]}, test pdg--{args.cross_type[1]}")
        tx, ts, ex, es, fs, t = load_crossdata(split_train=args.cross_type[0], split_test=args.cross_type[1])
    else:
        tx, ts, ex, es, fs, t = load_data(args.setsplit)

    tf, ef, ts, es = extract_features(tx, ex, ts, es, fs, t, args.fea_type)
    acc, f1, eer = clf_predict(tf, ef, ts, es, args.clf_type)
    
    print(f"用户分类准确率为{acc:.4f}, F1值为{f1:.4f}, 等错误率为{eer:.4f}")
    print("="*30)
    print(f"总用时{time.time()-start_time:.2f}s!")

    csv_path = os.path.join("/mnt/data1/tyl/UserID/csv", f"{args.setsplit}")
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    df = pd.DataFrame({"Acc": [acc], "F1": [f1], "EER": [eer]}).round(4)
    df.to_csv(os.path.join(csv_path, f"base_{args.clf_type}_{args.fea_type}.csv"))