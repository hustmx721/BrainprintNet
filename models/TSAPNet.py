import torch
import torch.nn as nn
import torch.nn.functional as F

def CalculateOutSize(model, Chans, Samples, nbands):
    '''
    Calculate the output based on input size.
    model is from nn.Module and inputSize is a array.
    '''
    device = next(model.parameters()).device
    x = torch.rand(1, nbands, Chans, Samples).to(device)
    out = model(x)
    return out.shape[-1]

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


class TSAPNet(nn.Module):

    def __init__(self, kernels, Samples,  temporalLayer = 'LogVarLayer', strideFactor= 4,
                    in_channels:int=22, nbands=12, num_classes=9, radix=8, fs=250):
        super(TSAPNet, self).__init__()
        self.kernels = kernels # type(List); conv_window kernels size
        self.parallel_conv = nn.ModuleList()
        self.strideFactor = strideFactor
        self.num_groups = len(self.kernels)
        self.Samples = Samples
        self.fs = fs
        
        # 1D-parallel_conv
        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv2d(in_channels=nbands, out_channels=nbands, kernel_size=(1,kernel_size),
                               stride=1, padding=0, bias=False, groups=nbands)
            self.parallel_conv.append(sep_conv)

        self.convblock = nn.Sequential(
            nn.BatchNorm2d(num_features=nbands),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=nbands, out_channels=nbands*radix, kernel_size=(in_channels,1),
                               stride=1, padding=0, bias=False))
        
        # self.temporalLayer = temporal_layer[temporalLayer](dim=-1)

        self.downsample = nn.Sequential(
            nn.BatchNorm2d(num_features=nbands*radix),
            nn.ReLU(inplace=False),
            nn.AvgPool2d((1,self.fs),(1,self.fs//2))
        )

        # calculate the input size of fc layer
        self.parallel_conv_out = self.num_groups * self.Samples - sum(k for k in self.kernels) + self.num_groups * 1
        self.fc_in = CalculateOutSize(model = nn.Sequential(self.convblock, self.downsample, nn.Flatten(start_dim=1)), 
                                      Chans=in_channels, Samples=self.parallel_conv_out, nbands=nbands)
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.fc_in, out_features=128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=num_classes),
            nn.LogSoftmax(dim=1))
        
    
    def _get_fea_dim(self,x):
        with torch.no_grad():
            features, _ = self.forward(x)
        return features.shape

    def forward(self, x):
        # 为了在dataloader时统一expand_dims方便, 这里检查维度
        x = x.squeeze(1) if x.dim() == 5 else x   # N, 1, nbands, C, T -> N, nbands, C, T
        out_sep = []
        # forward paralle 1D-conv blocks
        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)
        out = torch.cat(out_sep, dim=-1) # N, nbands, C, T'
        out = self.convblock(out)
        out = self.downsample(out)
        out = torch.flatten(out, start_dim=1)
        # out = torch.squeeze(out) # N, C', T'
        # pad_length = self.strideFactor - (out.shape[-1] % self.strideFactor)
        # if pad_length != 0:
        #     out = F.pad(out, (0, pad_length))

        # out = out.reshape([*out.shape[0:2], self.strideFactor, int(out.shape[-1]/self.strideFactor)])
        # out = self.temporalLayer(out)
        # out = torch.flatten(out, start_dim=1)

        features = out
        return features, self.fc(features)
    

# x = torch.rand(1, 1, 12, 62, 1000).to('cuda')
# model = TSAPNet(kernels=[11, 21, 31, 41, 51], Samples=1000, hidden_chans=32, temporalLayer='LogVarLayer', strideFactor=4,
#                     in_channels=62, nbands=12, num_classes=9, radix=8, fs=250).to('cuda')
# y = model(x)
# input()