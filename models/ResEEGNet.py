""" 
instruction:all versions of ResEEGNet
Author:hust-marx2
time: 2023/7/10
lastest:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# 
import math
import copy
from einops import rearrange
from einops import rearrange, reduce, repeat
# device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn import Linear


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


class ResEEGNet(nn.Module):

    def __init__(self, kernels, Samples, Resblock:int=2, hidden_chans:int=32, 
                    in_channels:int=22, fixed_kernel_size=5, num_classes=9):
        super(ResEEGNet, self).__init__()
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
        self.fc_in = self.planes * self.avgpool_out + 2 * 4 *self.planes # 卷积层输出和RNN输出拼接
        # self.fc_in = self.planes * self.avgpool_out  # 无RNN输出

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
        features = torch.cat([out, new_rnn_h], dim=1)  
        # features = out
        return features, self.fc(features)


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
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.25, inplace=False)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=(1,stride), padding=(0,padding), bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=(1,stride), padding=(0,padding), bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=(1,2), stride=(1,2))
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

# conv1 & conv2 --> depthwise conv 
# cause depthwise is 2D-conv, though the whole network is changed into 2D
# Don't forget to expand data dims from (C,T) to (1,C,T)
class ResEEGNet_v1(nn.Module):
    def __init__(self,
                 Samples, in_channels, num_classes, 
                 spatial_num : int = 2,
                 Resblock: int = 5,
                 kernels: list = [61,63,67,65,67,69],
                 fixed_kernel_size: int = 5,
                 hidden_chans:int = 32
                 ):
        super(ResEEGNet_v1,self).__init__()
        self.kernels = kernels # type(List); conv_window kernels size
        self.planes = hidden_chans # the kernels num of all the hidden layers(中间所有隐藏层通道数)
        self.in_channels = in_channels
        self.Samples = Samples
        self.D = spatial_num # 空域滤波器数量
        self.parallel_conv = nn.ModuleList()
        
        # 1D-parallel_conv
        # short = [3,5,7,9] ; long = [13,15,17,19]
        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv2d(in_channels=1, out_channels=self.planes, kernel_size=(1,kernel_size),
                               stride=1, padding=0, bias=False,)
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm2d(num_features=self.planes)
        self.relu = nn.ReLU(inplace=False)
        self.depthwise_conv = nn.Sequential(nn.Conv2d(in_channels = self.planes,
                      out_channels = self.planes * self.D,
                      kernel_size = (self.in_channels,1),
                      stride=(1,2),
                      groups = self.planes, # 分组卷积，聚合每个通道上信息
                      bias = False),
                      nn.Conv2d(in_channels=self.planes * self.D, out_channels=self.planes, kernel_size=(1,1),
                               stride=(1,1), padding=0, bias=False)
                               )
        
        # double Gem-Res Module
        self.block = self._make_resnet_layer(kernel_size=(1,fixed_kernel_size), stride=1, padding=fixed_kernel_size//2, blocks=Resblock)

        self.bn2 = nn.BatchNorm2d(num_features=self.planes)
        self.avgpool = nn.AvgPool2d(kernel_size=(1,8), stride=(1,8))

        # Calculate outpustsize for each block
        self.parallel_conv_out = len(self.kernels) * self.Samples - sum(k for k in self.kernels) + len(self.kernels) * 1
        self.conv1_out = self.parallel_conv_out // 2
        self.depthwise_conv_out = self.parallel_conv_out // 2 
        self.block_out = self.depthwise_conv_out // (2 ** Resblock)
        self.avgpool_out = self.block_out // 8
        # self.fc_in = self.planes * self.avgpool_out  # 无RNN输出
        self.fc_in = self.planes * self.avgpool_out + 2 * 4 *self.planes # 卷积层输出和RNN输出拼接

        self.rnn = nn.GRU(input_size=self.in_channels, hidden_size=4*self.planes, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(in_features=self.fc_in, out_features=num_classes)

    def _make_resnet_layer(self, kernel_size, stride, padding, blocks:int=2):
        layers = []

        for i in range(blocks):
            downsampling = nn.Sequential(
                    nn.AvgPool2d(kernel_size=(1,2), stride=(1,2))
                )
            layers.append(ResNet_Block(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
                                       stride=stride, padding=padding ,downsampling=downsampling))

        return nn.Sequential(*layers)
    
    def _get_fea_dim(self,x):
        with torch.no_grad():
            features, _ = self.forward(x)
        return features.shape

    def forward(self, x):
        # 为了在dataloader时统一expand_dims方便, 这里检查维度
        if x.dim() == 3:
            x = x.unsqueeze(1) # N, C, T -> N, 1, C, T
        elif x.dim() == 4:
            pass
        out_sep = []
        # forward paralle 1D-conv blocks  
        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)
        out = torch.cat(out_sep, dim=-1)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.depthwise_conv(out)
        out = self.block(out) # double Gem-Res Module
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)  
        out = out.reshape(out.shape[0], -1)  

        rnn_out, _ = self.rnn(x.squeeze().permute(0,2,1))
        new_rnn_h = rnn_out[:, -1, :]  
        features = torch.cat([out, new_rnn_h], dim=1) 

        # features = out
        return features, self.fc(features)


# we change the RNN structure at 
class ResEEGNet_v2(nn.Module):

    def __init__(self, kernels, Samples, Resblock:int=2, hidden_chans:int=32, in_channels:int=22, fixed_kernel_size=5, num_classes=9):
        super(ResEEGNet_v2, self).__init__()
        self.kernels = kernels # type(List); conv_window kernels size
        self.planes = hidden_chans # the channel num of all the hidden layers(中间所有隐藏层通道数)
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels
        self.Samples = Samples
        
        # 1D-parallel_conv
        # short = [3,5,7,9] ; long = [13,15,17,19]
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
        self.fc_in = 2 * self.planes **2

        self.rnn = nn.GRU(input_size=self.avgpool_out, hidden_size=self.planes, num_layers=2, bidirectional=True)

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
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.block(out) # double Gem-Res Module
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)  
        features, _ = self.rnn(out)
        features = features.reshape(features.shape[0],-1)
        return features, self.fc(features)




# iot = torch.randn(100,1,40,1000)#.cuda()
# model = ResEEGNet_v2(kernels=[11,21,31,41,51],Resblock=5, Samples=1000, in_channels=40, fixed_kernel_size=5, num_classes=9)#.cuda()
# fea, output = model(iot)
# print(fea.shape)
# print(model)
# torch.save(model,"/data2/tyl/baselines/UserId/visual/ResEEGNetv2.pth")
# model = torch.load("/data2/tyl/baselines/UserId/visual/base_model.pth")
# model.eval()
# traced_script_module = torch.jit.trace(model, iot)
# traced_script_module.save('/data2/tyl/baselines/UserId/visual/base_model.pt')#保存路径



# Bert模型构建
def get_attn_pad_mask(seq_q):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]

class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=61):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, dropout=0.):
        super().__init__()
        self.fc_1 = nn.Linear(d_model, ff_dim)
        self.fc_2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print('x',x.shape) #(64,2,64)
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


def clone_module(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])



class MultiHeadSelfAttention(nn.Module): #自注意力机制
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.proj_qkv = nn.Linear(embed_dim, embed_dim*3, bias=True)
        self.proj_out = nn.Sequential(
                            nn.Linear(embed_dim, embed_dim, bias=False),
                            nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        qkv = self.proj_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # print('q',q.shape)
        # print('k', k.shape)
        # print('v', v.shape)
        # q torch.Size([35, 4, 144, 32])
        # k torch.Size([35, 4, 144, 32])
        # v torch.Size([35, 4, 144, 32])
        product = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # product.masked_fill_(attn_mask.cuda(), -1e9)
        # print(product)
        # print('product',product.shape) # 35,4,48,48 可以加attention mask
        weights = F.softmax(product, dim=-1)
        weights_o = weights
        # weights = self.dropout(weights)

        out = torch.matmul(weights, v)
        # print('out',out.shape)  #64, 4, 99, 16]
        # combine heads
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.proj_out(out),weights_o


class SelfEncoderLayer(nn.Module): #重点看一看，很重要
    def __init__(self, embed_dim=64, hidden_dim=64, num_heads=4):
        super().__init__()
        self.Attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.LayerNorm = nn.LayerNorm(embed_dim)
        self.LayerNorm_attn = nn.LayerNorm(embed_dim)
        self.FeedForward = FeedForward(embed_dim, hidden_dim)
    def forward(self, x,attn_mask):
        norm_x = self.LayerNorm(x)
        attention,qk = self.Attention(norm_x,attn_mask)
        attn = attention + norm_x
        norm_attn = self.LayerNorm_attn(attn)
        output = self.FeedForward(norm_attn) + norm_attn
        return output
    
# self.SelfEncoder = SelfEncoderLayer(embed_dim=100, hidden_dim=100, num_heads=4)
# feature_ = self.SelfEncoder(feature_, get_attn_pad_mask(torch.ones(feature_.shape[0], feature_.shape[1])))