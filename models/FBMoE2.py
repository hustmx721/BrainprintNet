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

import torch
import torch.nn.functional as F
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

class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)    

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
    def __init__(self, input_size, output_size, hidden_size):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten()  
        self.fc1 = LinearWithConstraint(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = LinearWithConstraint(hidden_size, output_size)
        self.clf = nn.Sequential(self.flatten,self.fc1,self.relu,self.fc2)

    def forward(self, x):
        x = self.clf(x)
        return x


class MoE(nn.Module):
    def __init__(self, nbands, num_experts, m, emb_size, n_features):
        super(MoE, self).__init__()
        
        # 定义专家网络，每个专家由卷积和全连接层组成
        self.experts = nn.ModuleList([
            nn.Sequential(
                Conv1dWithConstraint(nbands, m*num_experts, kernel_size=1),
                swish(),
                nn.Linear(n_features, emb_size),
            ) for _ in range(num_experts)
        ])
        
        # 门控网络，决定每个专家的权重
        self.gate = nn.Linear(nbands, m*num_experts)
    
    def forward(self, x):
        gate_weights = F.softmax(self.gate(x.mean(dim=-1)), dim=-1)  # (batch_size, num_experts)
         # 初始化输出张量
        out = 0
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)  # (batch_size, m * nbands, emb_size)
            # 扩展 gate_weights 的形状，使其与 expert_out 兼容
            weight = gate_weights[:, i].view(-1, 1, 1)  # (batch_size, 1, 1)
            out += weight * expert_out  # 广播相乘，(batch_size, m * nbands, emb_size)
        # out = sum(w * expert(x) for w, expert in zip(gate_weights.T, self.experts))
        return out


class ConvBlocks(nn.Module):
    def __init__(self, kernels, nbands, m=8, dropout=0.5,
                 num_experts=8, emb_size=64, in_channels=22, fs=250):
        super(ConvBlocks, self).__init__()
        
        self.kernels = kernels
        self.parallel_conv = nn.ModuleList()
        
        # 2D-parallel_conv: 提取不同尺度的时域特征
        for i, kernel_size in enumerate(self.kernels):
            sep_conv = Conv2dWithConstraint(in_channels=nbands, out_channels=nbands, kernel_size=(1, kernel_size),
                                 groups=nbands, bias=False)
            self.parallel_conv.append(sep_conv)

        # 空间卷积，用于跨通道特征提取
        self.sconv = nn.Sequential(Conv2dWithConstraint(in_channels=nbands, out_channels=nbands, kernel_size=(in_channels, 1), 
                               groups=nbands, bias=False),
                               nn.BatchNorm2d(nbands), 
                               swish(),
                               nn.Dropout2d(dropout))
        
        # 全局池化，降低时域维度
        self.avgpool = nn.AdaptiveAvgPool2d((1, int(fs//2)))  
        # MoE模块：混合不同频带的信息
        self.moe = MoE(nbands=nbands, num_experts=num_experts, m=m, emb_size=emb_size, n_features=int(fs//2))
        
    def forward(self, x):
        # input tensor: (B, nbands, C, T)
        # 时空卷积
        B, nbands, C, T = x.shape
        conv_outputs = []
        for conv in self.parallel_conv:
            conv_output = conv(x)  # 每个kernel尺寸的卷积
            conv_outputs.append(conv_output)
        x = torch.cat(conv_outputs, dim=-1)
        x = self.sconv(x)  # 空间卷积
        x = torch.squeeze(self.avgpool(x))  # 全局池化, (B, nbands, fs//2)
        # MoE多频带混合信息
        moe_output = self.moe(x)  # (B, nbands*m, emb_size)
        
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



class MoETSformer2(nn.Module):
    def __init__(self, kernels, Chans, nbands, fs, n_classes, 
                 m=8, num_experts=8, emb_size=64, depth=4, **kwargs):
        super().__init__()
        
        self.fs = fs
        self.conv_embedding = ConvBlocks(kernels=kernels, in_channels=Chans, nbands=nbands, m=m,
                                         num_experts=num_experts, emb_size=emb_size, fs=fs)
        self.transformer_encoder = TransformerEncoder(depth=depth, emb_size=emb_size)
        self.classification_head = Classifier(input_size=m*nbands*emb_size , output_size=n_classes, hidden_size=emb_size)

    def forward(self, x):
        device = x.device
        # 为了在dataloader时统一expand_dims方便, 这里检查维度
        if x.dim() == 3:
            pass
        elif x.dim() == 4:
            x = x.squeeze(1)  # N, 1, C, T -> N, C, T
        x = SubBandSplitTensorFIR(x,8,32,4,self.fs)  # N, nBands, C, T

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
# nexperts = 6
# in_channels = 62
# fs = 250
# n_classes = 50
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = MoETSformer2(kernels=kernels, nbands=nbands, num_experts=nexperts, Chans=in_channels, fs=fs,
#                     n_classes=n_classes, emb_size=64, depth=1).to(device)

# input_data = torch.randn(64, 1, in_channels, Samples).to(device)  # 假设批次大小为8，22个通道，100个时间采样
# output, _ = model(input_data)
# print("输出形状:", output.shape)  # 期望输出形状为 (8, num_classes)

# data = torch.randn(1, 62, 1000)
# subdata = SubBandSplitTensorFIR(data, 8, 32, 4, 250)
# print(subdata.shape)