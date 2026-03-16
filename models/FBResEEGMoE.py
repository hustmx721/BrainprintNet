import torch
import torch.nn as nn
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


class MoE(nn.Module):
    def __init__(self, nbands, num_experts, m, poolsize, top_k=3):
        super(MoE, self).__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 定义专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(nbands, m * nbands, kernel_size=1),
                nn.AvgPool1d(kernel_size=poolsize, stride=poolsize),
                nn.BatchNorm1d(m * nbands),
                nn.ReLU(),
            ) for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate = nn.Linear(nbands, m * nbands)
    
    def forward(self, x):
        # 计算门控权重
        gate_weights = F.softmax(self.gate(x.mean(dim=-1)), dim=-1)  # (batch_size, num_experts)
        # 使用 detach 计算 mask
        with torch.no_grad():
            topk_values, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)
            mask = torch.zeros_like(gate_weights)
            mask.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_values))
        # 应用 mask, 归一化权重
        gate_weights = gate_weights * mask  # 非 top-k 的权重置为 0
        gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True)
        # 初始化输出张量
        out = 0
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            weight = gate_weights[:, i].view(-1, 1, 1)
            out += weight * expert_out
        
        return out


# baseline ResEEGNet
# para: kernels = [11, 21, 31, 41, 51]; 2-layers GRU
class FBResEEGMoE(nn.Module):

    def __init__(self, kernels, Samples, nbands:int=6, m:int=8, top_k=3, num_experts=6,
                    in_channels:int=22,  num_classes=9,  fs=250):
        super(FBResEEGMoE, self).__init__()
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
        
        self.moe = MoE(nbands, num_experts=num_experts, m=m, poolsize=fs//2, top_k=top_k)

        self.fc = nn.Sequential(nn.Linear(in_features=m*nbands*(Samples//(fs//2)), out_features=num_classes),
                nn.LogSoftmax(dim=1))

    
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
        out = torch.cat(out_sep, dim=-1) # N, nbands, C, T * len(self.kernels)
        out = self.avgpool(out)  # N, nbands, C, T
        out = self.sconv(out) # N, nbands, 1, T
        out = torch.squeeze(out)
        out = self.moe(out)  # N, m*nbands, T // (fs//2)
        out = out.reshape(out.size(0), -1)  

        return out, self.fc(out)

    
# x = torch.rand(64,1,62,1000)
# model = FBResEEG2D3(kernels=[11, 21, 31, 41, 51], Samples=1000,  nbands=6, m=8, 
#                     in_channels=62, num_classes=9,  fs=250)

# y, z = model(x)
# print(y.shape, z.shape)