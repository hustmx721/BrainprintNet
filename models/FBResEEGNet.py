import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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

    # return rearrange(sub_band_data, 'b c t n -> b (c n) t')
    return sub_band_data


def CalculateOutSize(model, Chans, Samples):
    '''
    Calculate the output based on input size.
    model is from nn.Module and inputSize is a array.
    '''
    device = next(model.parameters()).device
    x = torch.rand(1, 1, Chans, Samples).to(device)
    out = model(x)
    return out.shape[-1]

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
class FBResEEGNet_v1(nn.Module):

    def __init__(self, kernels, Samples, Resblock:int=2, hidden_chans:int=32, 
                    in_channels:int=22, fixed_kernel_size=5, num_classes=9, radix=6, fs=250):
        super(FBResEEGNet_v1, self).__init__()
        self.kernels = kernels # type(List); conv_window kernels size
        self.planes = hidden_chans # the channel num of all the hidden layers(中间所有隐藏层通道数)
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels * radix
        self.Samples = Samples
        self.fs = fs
        
        # 1D-parallel_conv
        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.planes, kernel_size=(kernel_size),
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
        self.fc_in = self.planes * self.avgpool_out + 2 * 4 *self.planes # 卷积层输出和RNN输出拼接
        # self.fc_in = self.planes * self.avgpool_out  # 无RNN输出
        # self.bn = nn.BatchNorm1d(num_features=2 * 4 *self.planes)
        self.rnn = nn.GRU(input_size=self.in_channels, hidden_size=4*self.planes, num_layers=2, bidirectional=True)
        # self.rnn = nn.LSTM(input_size=self.in_channels, hidden_size=4*self.planes, num_layers=2, bidirectional=True)
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
        device = x.device
        if x.dim() == 3:
            pass
        elif x.dim() == 4:
            x = x.squeeze(1)  # N, 1, C, T -> N, C, T
        out_sep = []
        x = SubBandSplit(x.cpu().numpy(),8,32,4,self.fs)
        x = torch.from_numpy(x).to(torch.float32).to(device)  # N, C, T, nBands
        x = rearrange(x, 'b c t n -> b (n c) t')
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

        rnn_out, _ = self.rnn(x.permute(0,2,1))
        new_rnn_h = rnn_out[:, -1, :]  
        # new_rnn_h = self.bn(new_rnn_h)
        features = torch.cat([out, new_rnn_h], dim=1)  
        # features = out
        return features, self.fc(features)
    