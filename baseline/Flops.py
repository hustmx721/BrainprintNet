import torch, sys
sys.path.append("/mnt/data1/tyl/UserID")
import time  # For inference time
from thop import profile  # For FLOPs
import pandas as pd

from models.MyModels import EEGNet,ShallowConvNet,DeepConvNet,CNN_LSTM
from models.EEGConformer import Conformer
from models.FBCNet import FBCNet
from models.FBMSNet import FBMSNet
from models.IFNet import IFNet
from models.modelV1 import ResEEGNet
from models.best2models import FBMSTSNet, MSNet
from frameworks.DARN.my_network import DARNNet
from frameworks.DRFNet.my_network import DRFNet
from frameworks.MCGP.MCGP import MCGPnet

results = dict()

batch_size = 1
nclass = 20
channel = 64
samples = 1000
fs = 250

input_tensors = torch.randn(1,1,64,1000)
fb_tensors = torch.randn(1,12,64,1000)
model_names = ['EEGNet', 'ShallowConvNet', 'DeepConvNet', 'CNN_LSTM', 'Conformer', 'ResEEGNet',  'MSNet', 'DARNNet', 'DRFNet', 'MCGPnet']
fb_models = ['IFNet', 'FBCNet', 'FBMSNet', 'FBMSTSNet']
device = torch.device("cuda:5")

def load_model(model_name:str):
    if model_name == "EEGNet":
        model = EEGNet(classes_num=nclass,Chans=channel,Samples=samples)
    elif model_name == "DeepConvNet":
        model = DeepConvNet(classes_num=nclass,Chans=channel,Samples=samples)
    elif model_name == "ShallowConvNet":
        model = ShallowConvNet(classes_num=nclass,Chans=channel,Samples=samples)
    elif model_name == "CNN_LSTM":
        model = CNN_LSTM(channels=channel, n_classes=nclass, time_points=samples, 
                         hidden_size=128, num_layers=2)
    elif model_name == "Conformer":
        model = Conformer(Chans=channel, Samples=samples, n_classes=nclass)
    elif model_name == "ResEEGNet":
        model = ResEEGNet(kernels=[11,21,31,41,51], num_classes=nclass, in_channels=channel,
                          Samples=samples, Resblock=5)
    elif model_name == "MSNet":
        model = MSNet(kernels=[11,21,31,41,51], num_classes=nclass, in_channels=channel,
                      hidden_chans=64)
    elif model_name == "DARNNet":
        model = DARNNet(device=device, input_size=(1, channel, samples), num_class=nclass)
    elif model_name == "DRFNet":
        model = DRFNet(device=device, input_size=(1, channel, samples), num_class=nclass)
    elif model_name == "MCGPnet":
        model = MCGPnet(device=device, input_size=(1, channel, samples), num_class=nclass,
                        sampling_rate=samples)
    if model_name in fb_models:
        if model_name == "IFNet":
            model = IFNet(in_planes=channel, out_planes=64, num_classes=nclass, kernel_size=63,
                        radix=2, patch_size=int(fs/2), time_points=samples, fs=fs)
        elif model_name == "FBCNet":
            model = FBCNet(nChan=channel, fs=fs, nClass=nclass, nBands=12)
        elif model_name == "FBMSNet":
            model = FBMSNet(nChan=channel, fs=fs, nClass=nclass, nTime=samples)
        elif model_name == "FBMSTSNet":
            model = FBMSTSNet(kernels=[11,21,31,41,51], fs=fs, num_classes=nclass,
                              in_channels=channel, nbands=12)
    model = model.to(device)
    return model
    

def cal_results(model:torch.nn.Module, input_tensor:torch.Tensor):
    model.eval()
    times = []
    for _ in range(10):
        input_tensor = input_tensor.to(device)
        model = model.to(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            _ = model(input_tensor)
        end_event.record()
        torch.cuda.synchronize()  # 确保同步
        times.append(start_event.elapsed_time(end_event))
    times = times[1:]  # 去掉第一次
    inference_time = sum(times) / len(times)     # ms
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = 2 * macs / 1e9   # GFLOPs
    params = params / 1e6  # M
    return flops, params, inference_time

for model_name in model_names:
    model = load_model(model_name)
    input_tensors = input_tensors.to(device)
    flops, params, inference_time = cal_results(model, input_tensors)
    results[model_name] = {'FLOPs (GFLOPs)': flops, 'Params (M)': params, 'Inference Time (ms)': inference_time}
    
for model_name in fb_models:
    model = load_model(model_name)
    if model_name == "IFNet":
        fb_tensors = torch.randn(1,2*64,1000).to(device)
    else:
        fb_tensors = torch.randn(1,12,64,1000).to(device)
    flops, params, inference_time = cal_results(model, fb_tensors)
    results[model_name] = {'FLOPs (GFLOPs)': flops, 'Params (M)': params, 'Inference Time (ms)': inference_time}


df = pd.DataFrame.from_dict(results, orient="index").reset_index().rename(columns={"index": "Model"})

# 选择需要的列：Model / Params / FLOPs / Inference Time
df_out = df[[
    "Model",
    "Params (M)",
    "FLOPs (GFLOPs)",
    "Inference Time (ms)"]].copy().round(4)

# 保存 CSV
csv_path = "./model_costs.csv"
df_out.to_csv(csv_path, index=False)

# 打印结果
print(df_out.to_string(index=False))
print(f"\nSaved CSV to: {csv_path}")
