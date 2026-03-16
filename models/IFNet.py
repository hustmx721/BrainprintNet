# --------------------------------------------------------
# IFNet
# Written by Jiaheng Wang
# --------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_



import numpy as np
import scipy.signal
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from einops import rearrange


def SubBandSplit(data: np.ndarray, freq_start: int = 4, freq_end: int = 40, radix: int = 9, fs: int = 250):
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

    subbands = np.arange(freq_start, freq_end + 1, int((freq_end - freq_start) // radix))
    with ThreadPoolExecutor() as executor:
        # 准备每个频带的参数
        band_args = [(data, low_freq, high_freq) 
                     for low_freq, high_freq in zip(subbands[:-1], subbands[1:])]
        # 并行处理每个频带
        results = list(executor.map(process_single_band, band_args))

    # 重塑结果以匹配所需的输出形状
    sub_band_data = np.stack(results, axis=1)

    # return rearrange(sub_band_data, 'b c t n -> b (c n) t')
    return sub_band_data


class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        nn.Module.__init__(self)
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class LogPowerLayer(nn.Module):
    def __init__(self, dim):
        super(LogPowerLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(torch.mean(x ** 2, dim=self.dim), 1e-4, 1e4))
        #return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=False), 1e-4, 1e4))


class InterFre(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        out = sum(x)
        out = F.gelu(out)
        return out


class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, doWeightNorm = True, max_norm=0.5, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=0.5, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class Stem(nn.Module):
    def __init__(self, in_planes, out_planes = 64, kernel_size = 63, patch_size = 125, radix = 2):
        nn.Module.__init__(self)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = out_planes * radix
        self.kernel_size = kernel_size
        self.radix = radix
        self.patch_size = patch_size

        self.sconv = Conv(nn.Conv1d(self.in_planes, self.mid_planes, 1, bias=False, groups = radix),
                          bn=nn.BatchNorm1d(self.mid_planes), activation=None)

        self.tconv = nn.ModuleList()
        for _ in range(self.radix):
            self.tconv.append(Conv(nn.Conv1d(self.out_planes, self.out_planes, kernel_size, 1, groups=self.out_planes, padding=kernel_size // 2, bias=False,),
                                   bn=nn.BatchNorm1d(self.out_planes), activation=None))
            if kernel_size == 63:
                kernel_size //= 2
            elif kernel_size == 61:
                kernel_size = kernel_size // 2 + 1

        self.interFre = InterFre()

        self.power = LogPowerLayer(dim=3)
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        N, C, T = x.shape
        out = self.sconv(x)

        out = torch.split(out, self.out_planes, dim=1)
        out = [m(x) for x, m in zip(out, self.tconv)]

        out = self.interFre(out)

        out = out.reshape(N, self.out_planes, T // self.patch_size, self.patch_size)
        out = self.power(out)
        out = self.dp(out)
        return out


class IFNet(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, radix, patch_size, time_points, num_classes, fs):
        r'''Interactive Frequency Convolutional Neural Network V2

        :param in_planes: Number of input EEG channels
        :param out_planes: Number of output feature dimensions
        :param kernel_size: Temporal convolution kernel size
        :param radix:   Number of input frequency bands
        :param patch_size: Temporal pooling size
        :param time_points: Input window length
        :param num_classes: Number of classes
        '''
        nn.Module.__init__(self)
        self.fs = fs
        self.radix = radix
        self.in_planes = in_planes * radix
        self.out_planes = out_planes
        self.stem = Stem(self.in_planes, self.out_planes, kernel_size, patch_size=patch_size, radix=radix)

        self.fc = nn.Sequential(
            LinearWithConstraint(out_planes * (time_points // patch_size), num_classes, doWeightNorm=True),
        )
        #print(f'fc layer feature dims:{self.fc[-1].weight.shape}')
        self.apply(self.initParms)

    def initParms(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        device = x.device
        if x.dim() == 3:
            pass
        elif x.dim() == 4:
            x = x.squeeze(1)  # N, 1, C, T -> N, C, T
        x1 = SubBandSplit(x.cpu().numpy(),8,16,1,self.fs)
        x2 = SubBandSplit(x.cpu().numpy(),16,32,1,self.fs)
        x = torch.cat([torch.from_numpy(x1),torch.from_numpy(x2)],dim=1).to(torch.float32).to(device)
        x = torch.squeeze(x)
        out = self.stem(x)
        return out, self.fc(out.flatten(1))
    
                                                                                                                                          
# if __name__ == '__main__':
#     model = IFNet(in_planes=62, out_planes=64, kernel_size=63, radix=2, patch_size=125, time_points=1000, num_classes=50, fs=250)

#     X = np.random.rand(10,62,1000)
#     X = torch.from_numpy(X).to(torch.float32)
#     y = model(X)
#     print(y.shape)

# # Model name
# _C.MODEL.NAME = 'IFNet'
# _C.MODEL.NUM_CLASSES = 4
# _C.MODEL.TIME_POINTS = int(_C.DATA.DUR * int(_C.DATA.FS / _C.DATA.RESAMPLE))
# _C.MODEL.IN_CHANS = 22
# _C.MODEL.PATCH_SIZE = 125   #temporal pooling size
# _C.MODEL.EMBED_DIMS = 64
# _C.MODEL.KERNEL_SIZE = 63
# _C.MODEL.RADIX = 2
    
# _C.DATA.FILTER_BANK = [(4, 16), (16, 40)]