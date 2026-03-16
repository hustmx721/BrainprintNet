import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from scipy.signal import firwin

def SubBandSplitTensorFIR(data, freq_start=4, freq_end=40, bandwidth=4, fs=250, num_taps=21):
    """
    num_traps: Length of the filter (number of coefficients, i.e. the filter order + 1). 
    numtaps must be odd if a passband includes the Nyquist frequency.
    经验计算公式: num_traps = 4 / ((freq_end - freq_start) / fs)
    使用预计算FIR滤波系数的子带切分函数
    data (batch, channel, time) -> sub_band_data (batch, (channel*nBands), time)
    """
    def design_fir_filter(num_taps, freq_low, freq_high, fs):
        """设计FIR滤波器并返回其系数"""
        return torch.tensor(firwin(num_taps, [freq_low, freq_high], pass_zero=False, fs=fs), dtype=torch.float32)

    def apply_fir_filter(data, fir_kernel):
        """将FIR滤波器的系数应用于数据"""
        channels = data.size(1)
        # 将 FIR 核扩展到与通道数匹配，并设置 groups=channels 进行分组卷积
        kernel = fir_kernel.view(1, 1, -1).expand(channels, -1, -1)
        return F.conv1d(data, kernel, padding='same', groups=channels)
    device = data.device
    subbands = torch.arange(freq_start, freq_end + 1, bandwidth)
    
    results = []
    for low_freq, high_freq in zip(subbands[:-1], subbands[1:]):
        fir_kernel = design_fir_filter(num_taps, low_freq.item(), high_freq.item(), fs).to(device)
        filtered_data = apply_fir_filter(data, fir_kernel)
        results.append(filtered_data)

    sub_band_data = torch.stack(results, dim=1) # (batch, nBands, channel, time)
    return sub_band_data.to(torch.float32)

def CalculateOutSize(model, Chans, Samples):
    '''
    Calculate the output based on input size.
    model is from nn.Module and inputSize is a array.
    '''
    device = next(model.parameters()).device
    x = torch.rand(1, 1, Chans, Samples).to(device)
    out = model(x)
    return out.shape[-1]

class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''

    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

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
        return self.clf(x)


class ResNet_Block(nn.Module):
    """ 
    - 2D-ResNet block for adapting to 2D output from depthwise conv 
    - input(n,C,T) --> output (n,C,T//2) 
    - only downsampling by pooling layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsampling):
        # The kernel_size,stride,padding,downsampling compments for Timepoints dim (just a 1-D value,not tuple)
        # In another words,this block don't change the channel dim
        super(ResNet_Block, self).__init__()
        self.feedconv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.25, inplace=False),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,kernel_size),
                               stride=(1,stride), padding=(0,padding), bias=False)
        )
        self.feedconv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.25, inplace=False),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,kernel_size),
                               stride=(1,stride), padding=(0,padding), bias=False)
        )
        self.avgpool = nn.AvgPool2d(kernel_size=(1,2), stride=(1,2))
        self.downsampling = downsampling

    def forward(self, x):
        out = self.feedconv1(x)
        out = self.feedconv2(out)
        out = self.avgpool(out)
        identity = self.downsampling(x)
        return out + identity



# baseline ResEEGNet
# para: kernels = [11, 21, 31, 41, 51]; 2-layers GRU
class FBResEEG2D4(nn.Module):

    def __init__(self, kernels, Samples, nbands:int=6, m:int=8, fixed_kernel_size:int=5,  Resblock:int=2,
                    in_channels:int=22,  num_classes=9,  fs=250):
        super(FBResEEG2D4, self).__init__()
        self.kernels = kernels  # type(List); conv_window kernels size
        self.planes = nbands * m  # the channel num of all the hidden layers(中间所有隐藏层通道数)
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels
        self.Samples = Samples
        self.fs = fs
        
        # 2D-parallel_conv
        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv2d(in_channels=nbands, out_channels=nbands, kernel_size=(1,kernel_size),
                               padding=(0,kernel_size//2), groups=nbands, bias=False,)
            self.parallel_conv.append(sep_conv)
        
        self.avgpool = nn.AvgPool2d(kernel_size=(1,len(self.kernels)), stride=(1,len(self.kernels)))

        self.sconv = nn.Sequential(
            nn.BatchNorm2d(num_features=nbands),
            nn.ReLU(),
            nn.Dropout(p=0.25, inplace=False),
            nn.Conv2d(in_channels=nbands, out_channels=nbands, kernel_size=(in_channels,1),
                               stride=(1,1), padding=(0,0), bias=False, groups=nbands)
        )
        
        self.freqconv = nn.Sequential(
            nn.Conv2d(in_channels=nbands, out_channels=nbands*m, kernel_size=(1,1),
                               stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(num_features=nbands*m),
            nn.ReLU(),
            nn.Dropout(p=0.25, inplace=False),
        )

        # double Gem-Res Module
        self.block = self._make_resnet_layer(kernel_size=fixed_kernel_size, stride=1, padding=fixed_kernel_size//2, blocks=Resblock)

        self.fc = nn.Sequential(nn.Linear(in_features=m*nbands*(Samples//(2**Resblock)), out_features=num_classes),
                nn.LogSoftmax(dim=1))

    def _make_resnet_layer(self, kernel_size, stride, padding, blocks:int=2):
        layers = []

        for i in range(blocks):
            downsampling = nn.Sequential(
                    nn.AvgPool2d(kernel_size=(1,2), stride=(1,2))
                )
            layers.append(ResNet_Block(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
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
        x = SubBandSplitTensorFIR(x,8,32,4,self.fs).to(device)  # N, nBands, C, T
        # forward paralle 2D-conv blocks
        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)
        out = torch.cat(out_sep, dim=-1)
        out = self.avgpool(out)  # N, nbands, C, T
        out = self.sconv(out)
        out = self.freqconv(out)  # N, C*m, 1, T
        out = self.block(out)  
        out = out.reshape(out.size(0), -1)  
        return out, self.fc(out)



# x = torch.rand(64,1,62,1000)
# model = FBResEEG2D3(kernels=[11, 21, 31, 41, 51], Samples=1000, nbands=6, m=8, 
#                     in_channels=62, num_classes=9,  fs=250, Resblock=5, fixed_kernel_size=5)

# y, z = model(x)
# print(y.shape, z.shape)