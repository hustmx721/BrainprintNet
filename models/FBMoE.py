import torch
import torch.nn as nn
import torch.nn.functional as F
# 
import math
import copy
from einops import rearrange, reduce, repeat
# device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn import Linear
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce
from torch.backends import cudnn

cudnn.benchmark = False
cudnn.deterministic = True


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


class MoE(nn.Module):
    def __init__(self, nbands, num_experts, hidden_dim, out_dim, n_features):
        super(MoE, self).__init__()
        
        # 定义专家网络，每个专家由卷积和全连接层组成
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(nbands, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Linear(n_features, out_dim),
                nn.ReLU()
            ) for _ in range(num_experts)
        ])
        
        # 门控网络，决定每个专家的权重
        self.gate = nn.Linear(nbands, num_experts)
    
    def forward(self, x):
        gate_weights = F.softmax(self.gate(x.mean(dim=-1)), dim=-1)  # (batch_size, num_experts)
        out = sum(w * expert(x) for w, expert in zip(gate_weights.T, self.experts))
        return out


class ConvBlocks(nn.Module):
    def __init__(self, kernels, nbands, 
                 num_experts=8, hidden_chans=32, in_channels=22, fs=250):
        super(ConvBlocks, self).__init__()
        
        self.kernels = kernels
        self.parallel_conv = nn.ModuleList()
        self.parallel_bn = nn.ModuleList()
        
        # 2D-parallel_conv: 提取不同尺度的时域特征
        for i, kernel_size in enumerate(self.kernels):
            sep_conv = nn.Conv2d(in_channels=nbands, out_channels=nbands, kernel_size=(1, kernel_size),
                                 groups=nbands, bias=False)
            self.parallel_conv.append(sep_conv)
            self.parallel_bn.append(nn.BatchNorm2d(nbands))

        # 空间卷积，用于跨通道特征提取
        self.sconv = nn.Conv2d(in_channels=nbands, out_channels=nbands, kernel_size=(in_channels, 1), 
                               groups=nbands, bias=False)
        
        # 全局池化，降低时域维度
        self.avgpool = nn.AdaptiveAvgPool2d((1, int(fs//2)))  
        # MoE模块：混合不同频带的信息
        self.moe = MoE(nbands=nbands, num_experts=num_experts, hidden_dim=hidden_chans, out_dim=hidden_chans * 2, n_features=int(fs//2))
        
    def forward(self, x):
        # input tensor: (B, nbands, C, T)
        # 时空卷积
        B, nbands, C, T = x.shape
        conv_outputs = []
        for conv, bn in zip(self.parallel_conv, self.parallel_bn):
            conv_output = conv(x)  # 每个kernel尺寸的卷积
            conv_output = bn(conv_output)  # BN
            conv_output = F.relu(conv_output)  # ReLU
            conv_outputs.append(conv_output)
        x = torch.cat(conv_outputs, dim=-1)
        x = F.relu(self.sconv(x))  # 空间卷积
        x = self.avgpool(x)  # 全局池化
        x = torch.squeeze(x) # (B, nbands, fs//2)
        # MoE多频带混合信息
        moe_output = self.moe(x)  # (B, hidden_chans * 2, fs//2)
        
        return moe_output


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=8,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])



class MoETSformer(nn.Module):
    def __init__(self, kernels, Chans, nbands, fs, n_classes,
                 num_experts=8, emb_size=64, depth=4, **kwargs):
        super().__init__()
        
        self.fs = fs
        self.conv_embedding = ConvBlocks(kernels=kernels, in_channels=Chans, nbands=nbands, 
                                         num_experts=num_experts, hidden_chans=(emb_size//2), fs=fs)
        self.transformer_encoder = TransformerEncoder(depth=depth, emb_size=emb_size)
        self.classification_head = Classifier(input_size=int(emb_size**2//2) , output_size=n_classes, hidden_size=emb_size)

    def forward(self, x):
        device = x.device
        # 为了在dataloader时统一expand_dims方便, 这里检查维度
        if x.dim() == 3:
            pass
        elif x.dim() == 4:
            x = x.squeeze(1)  # N, 1, C, T -> N, C, T
        x = SubBandSplit(x.cpu().numpy(),8,32,4,self.fs)
        x = torch.from_numpy(x).to(torch.float32).to(device)  # N, C, T, nBands
        x = x.permute(0, 3, 1, 2)  # N, nBands, c, T
        x = self.conv_embedding(x)
        x = self.transformer_encoder(x)
        features = x.contiguous().view(x.size(0), -1)
        return features, self.classification_head(features)

    def _get_fea_dim(self, x):
        with torch.no_grad():
            features, _ = self.forward(x)
        return features.shape


# # 示例用法
# kernels = [7, 15, 31, 63, 127]
# Samples = 1500
# nbands = 6
# nexperts = 8
# in_channels = 62
# fs = 250
# n_classes = 50
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = MoETSformer(kernels=kernels, nbands=nbands, num_experts=nexperts, Chans=in_channels, fs=fs,
#                     n_classes=n_classes, emb_size=64, depth=4).to(device)

# input_data = torch.randn(64, nbands, in_channels, Samples).to(device)  # 假设批次大小为8，22个通道，100个时间采样
# output, _ = model(input_data)
# print("输出形状:", output.shape)  # 期望输出形状为 (8, num_classes)