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
def use_DWTA_(X, y):
    unique_labels = np.unique(y)
    X_aug_list = []
    y_aug_list = []
    
    for label in unique_labels:
        class_indices = np.where(y == label)[0]
        X_class = X[class_indices]
        
        # 生成所有可能的样本对，并对每对样本进行 DWTA 数据增强
        for i, j in combinations(range(len(X_class)), 2):
            # print(i,j)
            Xs = X_class[i]
            X_tar_train = X_class[j]
            Xs_aug, Xt_aug = DWTA(Xs, X_tar_train)
            
            X_aug_list.append(Xs_aug)
            y_aug_list.append(label)
            X_aug_list.append(Xt_aug)
            y_aug_list.append(label)
    
    X_aug = np.array(X_aug_list)
    y_aug = np.array(y_aug_list)
    
    return X_aug, y_aug


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


if __name__ == '__main__':
    N = 10 
    C = 3   
    T = 100 

    X = np.random.rand(N, C, T)
    # y = np.random.randint(0, 3, size=N)  # 假设有 3 类标签
    y = np.array([0,0,0,1,1,1,1,1,2,2])

    X_aug, y_aug = use_DWTA(X, y) # NOTE: 只返回增强后的数据

    print("Original X shape:", X.shape)
    print("Original y shape:", y.shape)
    print("Augmented X shape:", X_aug.shape)
    print("Augmented y shape:", y_aug.shape)
    print(y)
    print(y_aug)