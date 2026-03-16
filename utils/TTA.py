import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def Zscore(x):
    m = x.mean()
    d = x.std()
    return (x-m)/d


def Entropy(predict):
    epsilon = 1e-5
    H = -predict * torch.log(predict + epsilon)
    H = H.sum(dim=1)
    return H



def set_bn_eval_others_train(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.train()  # 设置 BN 层为训练模式
            for param in module.parameters():
                # 只更新运行的统计参数,不更新训练参数(实际上不反向传播也不会更新训练参数)
                param.requires_grad = True  
                module.weight.requires_grad = False  
                module.bias.requires_grad = False
        else:
            module.eval()  # 其他层设为评估模式
            for param in module.parameters():
                param.requires_grad = False  # 冻结其他层的参数


class IDTTA:
    def __init__(self, model, device):
        self.model = model

class RFTTA:
    def __init__(self, model, device):
        self.model = model