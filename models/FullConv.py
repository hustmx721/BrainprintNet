import torch
import torch.nn as nn
import torch.nn.functional as F
# 
import math
import copy
from einops import rearrange, reduce, repeat
# device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn import Linear


# two layers MLP Classifier
class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size:int=50):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten()  
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.clf = nn.Sequential(self.flatten,self.fc1,self.relu,self.fc2)

    def forward(self, x):
        x = self.clf(x)
        return x

class ResNet_1D_Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsampling):
        super(ResNet_1D_Block, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.25, inplace=False)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.downsampling = downsampling

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.avgpool(out)
        identity = self.downsampling(x)

        out += identity
        return out


# baseline ResEEGNet
# para: kernels = [11, 21, 31, 41, 51]; 2-layers GRU
class FullConv(nn.Module):

    def __init__(self, kernels, Samples, Resblock:int=2, hidden_chans:int=32, 
                    in_channels:int=22, fixed_kernel_size=5, num_classes=9):
        super(FullConv, self).__init__()
        self.kernels = kernels # type(List); conv_window kernels size
        self.planes = hidden_chans # the channel num of all the hidden layers(中间所有隐藏层通道数)
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels
        self.Samples = Samples
        
        # 1D-parallel_conv
        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(in_channels=in_channels, out_channels=self.planes, kernel_size=(kernel_size),
                               stride=1, padding=0, bias=False,)
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(in_channels=self.planes, out_channels=self.planes, kernel_size=fixed_kernel_size,
                               stride=2, padding=2, bias=False)
        
        # double Gem-Res Module
        self.block = self._make_resnet_layer(kernel_size=fixed_kernel_size, stride=1, padding=fixed_kernel_size//2, blocks=Resblock)

        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=8, stride=8)

        # Calculate outpustsize for each block
        self.parallel_conv_out = len(self.kernels) * self.Samples - sum(k for k in self.kernels) + len(self.kernels) * 1
        self.conv1_out = self.parallel_conv_out // 2
        self.block_out = self.conv1_out // (2 ** Resblock)
        self.avgpool_out = self.block_out // 8
        self.fc_in = self.planes * self.avgpool_out  # 无RNN输出
        self.fc = nn.Linear(in_features=self.fc_in, out_features=num_classes)

    def _make_resnet_layer(self, kernel_size, stride, padding, blocks:int=2):
        layers = []

        for i in range(blocks):
            downsampling = nn.Sequential(
                    nn.AvgPool1d(kernel_size=2, stride=2)
                )
            layers.append(ResNet_1D_Block(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsampling=downsampling))

        return nn.Sequential(*layers)
    
    def _get_fea_dim(self,x):
        with torch.no_grad():
            features, _ = self.forward(x)
        return features.shape

    def forward(self, x):
        # 为了在dataloader时统一expand_dims方便, 这里检查维度
        if x.dim() == 3:
            pass
        elif x.dim() == 4:
            x = x.squeeze(1)  # N, 1, C, T -> N, C, T
        out_sep = []
        # forward paralle 1D-conv blocks
        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)
        out = torch.cat(out_sep, dim=-1)
        # print("shape of tensor: ", out.shape)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.block(out) # double Gem-Res Module
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)  
        out = out.reshape(out.shape[0], -1)  

        features = out
        return features, self.fc(features)