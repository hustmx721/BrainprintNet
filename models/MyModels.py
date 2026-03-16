""" 
instruction:model of eegnet
Author:hust-marx2
time: 2023/7/10
lastest:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def CalculateOutSize(model, Chans, Samples):
    '''
    Calculate the output based on input size.
    model is from nn.Module and inputSize is a array.
    '''
    model.eval()
    device = next(model.parameters()).device
    x = torch.rand(1, 1, Chans, Samples).to(device)
    out = model(x)
    return out.shape[-1]

def LoadModel(model_name, Chans, Samples):
    if model_name == 'EEGNet':
        model = EEGNet(Chans=Chans,
                       Samples=Samples,
                       kernLenght=64,
                       F1=4,
                       D=2,
                       F2=8,
                       dropoutRate=0.25)
    elif model_name == 'DeepCNN':
        model = DeepConvNet(Chans=Chans, Samples=Samples, dropoutRate=0.5)
    elif model_name == 'ShallowCNN':
        model = ShallowConvNet(Chans=Chans, Samples=Samples, dropoutRate=0.5)
    else:
        raise 'No such model'
    return model


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

class EEGNet(nn.Module):
    """
    paras:
    samples: 采样点数量; kernel_size: 脑电信号中卷积核大小,对应原论文中的(1,64)卷积核;
    f1: 时域滤波器数量; f2: 点积滤波器数量; D: 空域滤波器数量;
    drop_out: we use p = 0.5 for within-subject classification and p = 0.25 for cross-subject classification
    """
    def __init__(self,
                 classes_num: int,
                 Chans: int, # 输入信号的通道维数，也即C
                 Samples: int, # 一般来说,不同任务也就前三个变量取不同值
                 kernel_size: int = 64,
                 f1: int = 8,
                 f2: int = 16,
                 D: int = 2,
                 drop_out: Optional[float] = 0.5):
        super(EEGNet, self).__init__()
        self.classes_num = classes_num
        self.in_channels = Chans
        self.samples = Samples
        self.kernel_size = kernel_size
        self.f1 = f1
        self.f2 = f2
        self.D = D
        self.drop_out = drop_out

        # time-conv2d,aggregate the temporal information
        # (1,C,T) --> (f1,C,T) ,上采样
        self.block1 = nn.Sequential(
            # four directions:left, right, up, bottom ;参数一般是默认(31,32,0,0)
            nn.ZeroPad2d((self.kernel_size // 2 - 1,
                          self.kernel_size - self.kernel_size // 2, 0,0)),  
            nn.Conv2d(in_channels = 1,
                      out_channels = self.f1,
                      kernel_size = (1,self.kernel_size), # 一般是默认值(1,64)
                      stride = 1,
                      bias = False ),# conv后若接norm层,bias设置为False,降低算力开销
            nn.BatchNorm2d(num_features = self.f1))  
    
        # DepthwiseConv2D,aggregate the spatial infomation
        # f1*C*T -conv--> (D*f1,1,T) -avgpool--> (D*f1,1,T//4) 
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = self.f1,
                      out_channels = self.f1 * self.D,
                      kernel_size = (self.in_channels,1),
                      groups = self.f1, # 分组卷积，聚合每个通道上信息
                      bias = False),
            nn.BatchNorm2d(num_features = self.f1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(self.drop_out))

        # depth-separable conv = Depthwise Convolution + Pointwise Convolution
        # point-conv for aggregate the info from all channels and change the num of channels
        # (D*f1,1,T//4) -seperableconv--> (f2,1,T//4) -avgpool--> (f2,1,T//32)
        self.block3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels = self.f1 * self.D, # depthwise-conv
                      out_channels = self.f1 * self.D,
                      kernel_size = (1, 16),
                      groups = self.f1 * self.D,
                      bias = False),
            nn.BatchNorm2d(num_features = self.f1 * self.D),
            nn.Conv2d(in_channels = self.f1 * self.D, # pointwise-conv
                      out_channels = self.f2,
                      kernel_size = (1, 1),
                      bias = False),
            nn.BatchNorm2d(num_features = self.f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.drop_out))
        
        # brainprint UsrId classifier
        self.clf = Classifier(input_size=self.f2 * (self.samples // (4 * 8)), output_size=self.classes_num)

        # task related classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features = self.f2 * (self.samples // (4 * 8)),
                      out_features = self.classes_num))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x) 
        x = self.block2(x)
        x = self.block3(x) 
        x = x.view(x.size(0),-1)
        # out = self.classifier(x)
        out = self.clf(x)
        return x, out
    
    # 网络输出预测熵
    def pred_ent(self,x):
        logits = self(x)
        lsm = nn.LogSoftmax(dim=-1)
        log_probs = lsm(logits)
        probs = torch.exp(log_probs)
        p_log_p = log_probs * probs
        predictive_entropy = -p_log_p.sum(axis=1)
        return predictive_entropy


class DeepConvNet(nn.Module):
    def __init__(self,
                 Chans: int,
                 Samples: int,
                 classes_num : int,
                 dropoutRate: Optional[float] = 0.5,
                 d1: Optional[int] = 25,
                 d2: Optional[int] = 50,
                 d3: Optional[int] = 100):
        super(DeepConvNet, self).__init__()

        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.classes_num = classes_num

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=d1, kernel_size=(1, 5)),
            nn.Conv2d(in_channels=d1, out_channels=d1, kernel_size=(Chans, 1)),
            nn.BatchNorm2d(num_features=d1), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=d1, out_channels=d2, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=d2), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=d2, out_channels=d3, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=d3), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.convT = CalculateOutSize(nn.Sequential(self.block1,self.block2,self.block3),self.Chans,self.Samples)
        self.clf = Classifier(input_size=self.convT*d3, output_size=self.classes_num) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = output.reshape(output.size(0), -1)
        out = self.clf(output)
        return output, out

    def MaxNormConstraint(self):
        for block in [self.block1, self.block2, self.block3]:
            for n, p in block.named_parameters():
                if hasattr(n, 'weight') and (
                        not n.__class__.__name__.startswith('BatchNorm')):
                    p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)


class Activation(nn.Module):
    def __init__(self, type):
        super(Activation, self).__init__()
        self.type = type

    def forward(self, input):
        if self.type == 'square':
            output = input * input
        elif self.type == 'log':
            output = torch.log(torch.clamp(input, min=1e-6))
        else:
            raise Exception('Invalid type !')

        return output


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
        self.clf = Classifier(input_size=self.convT*midDim, output_size=self.classes_num) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = output.reshape(output.size(0), -1)
        out = self.clf(output)
        return output, out

    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if hasattr(n, 'weight') and (
                    not n.__class__.__name__.startswith('BatchNorm')):
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)

    
class DomainDiscriminator(nn.Module):
    """
    Domain discriminator module - 2 layers MLP

    Parameters:
        - input_dim (int): dim of input features
        - hidden_dim (int): dim of hidden features
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int):
        super(DomainDiscriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
    

class Discriminator(nn.Module):
    def __init__(self, input_dim, n_subjects):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim
        self.n_subjects = n_subjects

        self.block = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=50, bias=True),
            nn.Linear(in_features=50, out_features=self.n_subjects, bias=True))

    def forward(self, feature):
        output = self.block(feature)

        return output
    
class CNN_LSTM(nn.Module):
    def __init__(self,channels,time_points,hidden_size,n_classes,num_layers,spatial_num=32,drop_out=0.25):
        super(CNN_LSTM, self).__init__()

        self.channels = channels
        self.time_points = time_points
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.drop_out = drop_out  
        self.spatial_num = spatial_num
        self.num_layers = num_layers

        self.block1 = nn.Sequential(
            nn.Conv2d(1,self.spatial_num,(self.channels,1),bias=False),
            nn.BatchNorm2d(self.spatial_num),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(self.drop_out))
        self.block2 = nn.Sequential(
            nn.Conv2d(self.spatial_num,2*self.spatial_num,(1,1),bias=False),
            nn.BatchNorm2d(2*self.spatial_num),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(self.drop_out))
        self.block3 = nn.Sequential(
            nn.Conv2d(2*self.spatial_num,4*self.spatial_num,(1,1),bias=False),
            nn.BatchNorm2d(4*self.spatial_num),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(self.drop_out))
        self.convblock = nn.Sequential(self.block1,self.block2,self.block3)
        self.convT = CalculateOutSize(self.convblock,self.channels,self.time_points)
        
        self.lstm = nn.LSTM(8*self.convT, self.hidden_size, self.num_layers, batch_first=True)
        self.clf = nn.Sequential(nn.Linear(in_features=self.hidden_size*self.spatial_num//2, out_features=self.hidden_size),
                                 nn.Linear(in_features=self.hidden_size, out_features=self.n_classes))
        
    
    def forward(self,x:torch.Tensor)-> torch.Tensor:
        # X (batch_size,1,channels,time_points) = (B,1,C,T)
        x = self.block1(x) # (B,spatial_num,1,T//2)
        x = self.block2(x) # (B,2*spatial_num,1,T//4)
        x = self.block3(x) # (B,4*spatial_num,1,T//8)  eg:(32,128,1,125)
        x = x.reshape(x.shape[0],-1,8*self.convT) # (32,16,1000) # (B,spatial_num//2,T) 
        x, _ = self.lstm(x) # (B,spatial_num//2,hidden_size) eg:(32,16,192)
        x = x.reshape(x.shape[0],-1)
        return x, self.clf(x)

class MI_CNN(nn.Module):
    def __init__(self,num_classes:int=9):
        super(MI_CNN, self).__init__()

        self.L1M1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2) )        
        self.L2M2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2))        
        self.L3M2R1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(5,128)),
            nn.ReLU())       
        self.L4M2R1 = nn.Linear(in_features=1024, out_features=num_classes)
        
    def forward(self, x): # (1,22,512)
        x = self.L1M1(x)  # (256,11,256)
        x = self.L2M2(x) # (512,5,128)
        x = self.L3M2R1(x) # (1024,1,1)
        x = torch.squeeze(x)
        x = self.L4M2R1(x)
        return x
    
class CNN_RNN(nn.Module):
    def __init__(self,channel,time_point,hidden_size,num_layer,spatial_num=32,n_classes=9,drop_out=0.25):
        super(CNN_RNN, self).__init__()
        self.chan = channel
        self.spa_num = spatial_num
        self.ncls = n_classes
        self.dp = drop_out
        self.time_point = time_point
        self.nlayer = num_layer
        self.hidsize = hidden_size

        self.block1 = nn.Sequential(
            nn.Conv2d(1,self.spa_num,(self.chan,1),bias=False),
            nn.BatchNorm2d(self.spa_num),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(self.dp))
        self.block2 = nn.Sequential(
            nn.Conv2d(self.spa_num,2*self.spa_num,(1,1),bias=False),
            nn.BatchNorm2d(2*self.spa_num),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(self.dp))
        self.block3 = nn.Sequential(
            nn.Conv2d(2*self.spa_num,4*self.spa_num,(1,1),bias=False),
            nn.BatchNorm2d(4*self.spa_num),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(self.dp))
        self.convblock = nn.Sequential(self.block1,self.block2,self.block3)
        self.convT = CalculateOutSize(self.convblock,self.chan,self.time_point)
        
        self.rnn = nn.RNN(input_size=self.convT*8, hidden_size=self.hidsize, num_layers=self.nlayer, batch_first=True)
        self.clf = nn.Sequential(nn.Linear(in_features=self.hidsize*self.spa_num//2, out_features=self.hidsize),
                                 nn.Linear(in_features=self.hidsize, out_features=self.ncls))