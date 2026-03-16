import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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

        rnn_out, _ = self.rnn(x.permute(0,2,1))
        new_rnn_h = rnn_out[:, -1, :]  
        # new_rnn_h = self.bn(new_rnn_h)
        features = torch.cat([out, new_rnn_h], dim=1)  
        # features = out
        return features, self.fc(features)
    

class ShallowConvNet(nn.Module):
    def __init__(
        self,
        classes_num: int,
        Chans: int,
        Samples: int,
        dropoutRate: Optional[float] = 0.5, midDim: Optional[int] = 40,
    ):
        super(ShallowConvNet, self).__init__()

        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.classes_num = classes_num

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=midDim, kernel_size=(1, 13)),
            nn.Conv2d(in_channels=midDim,
                      out_channels=midDim,
                      kernel_size=(self.Chans, 1)),
            nn.BatchNorm2d(num_features=midDim), 
            nn.ELU(), #Activation('square'),
            nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7)),
            nn.ELU(), # Activation('log'), 
            nn.Dropout(self.dropoutRate))
        self.convT = CalculateOutSize(self.block1,self.Chans,self.Samples)
        self.rnn = nn.GRU(input_size=self.Chans, hidden_size=128, num_layers=2, bidirectional=True)
        self.clf = Classifier(input_size=self.convT*midDim+2*128, output_size=self.classes_num) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = output.reshape(output.size(0), -1)
        rnn_out, _ = self.rnn(x.squeeze(axis=1).permute(0,2,1))
        new_rnn_h = rnn_out[:, -1, :]
        features = torch.cat([output, new_rnn_h], dim=-1)  
        logits = self.clf(features)
        return features, logits

    def _get_fea_dim(self,x):
        with torch.no_grad():
            features, _ = self.forward(x)
        return features.shape
    
    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if hasattr(n, 'weight') and (
                    not n.__class__.__name__.startswith('BatchNorm')):
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)


import math
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch.backends import cudnn

cudnn.benchmark = False
cudnn.deterministic = True

# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, kernels, Chans, Samples, emb_size=32):
        # self.patch_size = patch_size
        super().__init__()
        self.kernels = kernels
        self.Chans = Chans
        self.Samples = Samples  
        self.parallel_conv = nn.ModuleList()
        self.parallel_conv_out = len(self.kernels) * self.Samples - sum(k for k in self.kernels) + len(self.kernels) * 1
        for kernel_size in self.kernels:
            self.parallel_conv.append(nn.Conv2d(in_channels=1, out_channels=emb_size, kernel_size=(1,kernel_size), stride=1, padding=0,bias=False))
        
        self.shallownet = nn.Sequential(
            nn.Conv2d(in_channels=emb_size,
                      out_channels=emb_size,
                      kernel_size=(self.Chans, 1)),
            nn.BatchNorm2d(num_features=emb_size), 
            nn.ELU(), #Activation('square'),
            nn.AvgPool2d(kernel_size=(1, 110), stride=(1, 22)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.ELU(), # Activation('log'), 
            nn.Dropout(0.5))

        self.projection = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        out_sep = []
        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)
        out = torch.cat(out_sep, dim=-1)
        out = self.shallownet(out)
        out = self.projection(out)
        return out # (bs, 46, 32)


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


class ClassificationHead(nn.Sequential):
    def __init__(self, kernels, Chans, Samples, depth, emb_size, n_classes, fc_in):
        super().__init__()
        
        # global average pooling
        self.fc_in = fc_in
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        out = self.fc(x)
        return x, out
    
    def _get_fea_dim(self,x):
        with torch.no_grad():
            features, _ = self.forward(x)
        return features.shape


    
class MyConformer(nn.Module):
    def __init__(self, kernels, Chans, Samples, emb_size=32, depth=4, n_classes=4, **kwargs):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(kernels, Chans, Samples, emb_size)
        self.para_conv_out = self.patch_embedding.parallel_conv_out
        self.fc_in = int((self.para_conv_out-110) // 22 + 1) * emb_size
        self.transformer_encoder = TransformerEncoder(depth, emb_size)
        self.classification_head = ClassificationHead(kernels, Chans, Samples, depth, emb_size, n_classes,self.fc_in)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        features = x.contiguous().view(x.size(0), -1)
        features, out = self.classification_head(features)
        return features, out

    def _get_fea_dim(self, x):
        with torch.no_grad():
            features, _ = self.forward(x)
        return features.shape
# base版: emb_size=32, depth=4
# middle版: emb_size=64, depth=4 (对号)
# large版: emb_size=64, depth=6

# iot = torch.randn(100,1,40,1000)#.cuda()
# model = ResEEGNet(kernels=[11,21,31,41,51],Resblock=5, Samples=1000, in_channels=40, fixed_kernel_size=5, num_classes=9)#.cuda()
# fea, output = model(iot)
# print(fea.shape)
# print(model)