import torch
import torch.nn as nn
import torch.nn.functional as F

#%% Support classes for FBNet Implementation
class VarLayer(nn.Module):
    '''
    The variance layer: calculates the variance of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim = self.dim, keepdim= True)

class StdLayer(nn.Module):
    '''
    The standard deviation layer: calculates the std of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(StdLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.std(dim = self.dim, keepdim=True)

class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))

class MeanLayer(nn.Module):
    '''
    The mean layer: calculates the mean of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MeanLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim = self.dim, keepdim=True)

class MaxLayer(nn.Module):
    '''
    The max layer: calculates the max of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MaxLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        ma ,ima = x.max(dim = self.dim, keepdim=True)
        return ma

temporal_layer = {'VarLayer': VarLayer, 'StdLayer': StdLayer, 'LogVarLayer': LogVarLayer, 'MeanLayer': MeanLayer, 'MaxLayer': MaxLayer}

# baseline ResEEGNet
# para: kernels = [11, 21, 31, 41, 51]; 2-layers GRU
class DBNet3(nn.Module):

    def __init__(self, kernels, Samples, hidden_chans:int=32, temporalLayer = 'LogVarLayer', strideFactor= 4,
                    in_channels:int=22,num_classes=9,fs=250):
        super(DBNet3, self).__init__()
        self.kernels = kernels # type(List); conv_window kernels size
        self.planes = hidden_chans # the channel num of all the hidden layers(中间所有隐藏层通道数)
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels
        self.Samples = Samples
        self.fs = fs
        self.strideFactor = strideFactor
        
        # 1D-parallel_conv
        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=(kernel_size),
                               stride=1, padding=0, bias=False, groups=self.in_channels)
            self.parallel_conv.append(sep_conv)
        
        self.convblock = nn.Sequential(
            nn.BatchNorm1d(num_features=self.in_channels),
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.planes, kernel_size=1,
                               stride=1, padding=0, bias=False),
            nn.BatchNorm1d(num_features=self.planes),
            nn.ReLU(inplace=False)
                               )
        self.temporalLayer = temporal_layer[temporalLayer](dim=-1)

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.planes*strideFactor, out_features=num_classes),
            nn.LogSoftmax(dim=1))
    
    def _get_fea_dim(self,x):
        with torch.no_grad():
            features, _ = self.forward(x)
        return features.shape

    def forward(self, x):
        # 为了在dataloader时统一expand_dims方便, 这里检查维度
        x = torch.squeeze(x)
        out_sep = []
        # forward paralle 1D-conv blocks
        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)
        out = torch.cat(out_sep, dim=-1)
        out = self.convblock(out)

        pad_length = self.strideFactor - (out.shape[-1] % self.strideFactor)
        if pad_length != 0:
            out = F.pad(out, (0, pad_length))

        out = out.reshape([*out.shape[0:2], self.strideFactor, int(out.shape[-1]/self.strideFactor)])
        out = self.temporalLayer(out)
        out = torch.flatten(out, start_dim=1)

        features = out
        return features, self.fc(features)
    


class DBNet3_1(nn.Module):

    def __init__(self, kernels, Samples, hidden_chans:int=32, temporalLayer = 'LogVarLayer', strideFactor= 4,
                    in_channels:int=22,num_classes=9,fs=250):
        super(DBNet3_1, self).__init__()
        self.kernels = kernels # type(List); conv_window kernels size
        self.planes = hidden_chans # the channel num of all the hidden layers(中间所有隐藏层通道数)
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels
        self.Samples = Samples
        self.fs = fs
        self.strideFactor = strideFactor
        
        # 1D-parallel_conv
        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=(kernel_size),
                               stride=1, padding=kernel_size//2, bias=False, groups=self.in_channels)
            self.parallel_conv.append(sep_conv)
        
        self.convblock = nn.Sequential(
            nn.BatchNorm1d(num_features=self.in_channels),
            nn.ReLU(inplace=False),
            nn.AvgPool1d(kernel_size=len((self.kernels)), stride=len((self.kernels))),
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.planes, kernel_size=5,
                               stride=2, padding=2, bias=False),
            nn.BatchNorm1d(num_features=self.planes),
            nn.ReLU(inplace=False)
                               )
        self.temporalLayer = temporal_layer[temporalLayer](dim=-1)

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.planes*strideFactor, out_features=num_classes),
            nn.LogSoftmax(dim=1))
    
    def _get_fea_dim(self,x):
        with torch.no_grad():
            features, _ = self.forward(x)
        return features.shape

    def forward(self, x):
        # 为了在dataloader时统一expand_dims方便, 这里检查维度
        x = torch.squeeze(x)
        out_sep = []
        # forward paralle 1D-conv blocks
        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)
        out = torch.cat(out_sep, dim=-1)
        out = self.convblock(out)

        pad_length = self.strideFactor - (out.shape[-1] % self.strideFactor)
        if pad_length != 0:
            out = F.pad(out, (0, pad_length))

        out = out.reshape([*out.shape[0:2], self.strideFactor, int(out.shape[-1]/self.strideFactor)])
        out = self.temporalLayer(out)
        out = torch.flatten(out, start_dim=1)

        features = out
        return features, self.fc(features)