# EEGNet, DeepConvNet, ShallowConvNet, Conformer, FBCNet, MyConformerWithConstraint, ResEEGNet, CNN_LSTM
# "001","004","BCI85","Rest85", "M3CV_Rest", OpenBMI:"Rest", "MI", "ERP", "SSVEP", "cross"
# "M3CV_Rest", "M3CV_Transient", "M3CV_Steady", "M3CV_P300", "M3CV_Motor", "M3CV_SSVEP_SA", "SEED"
python -u main.py --setsplit=001 --gpuid=0 --model=ResEEGNet --ctr_loss=True --alpha=1e-4
python -u main.py --setsplit=004 --gpuid=0 --model=ResEEGNet --ctr_loss=True --alpha=1e-4  
python -u main.py --setsplit=ERP --gpuid=0 --model=ResEEGNet --ctr_loss=True --alpha=1e-4  
python -u main.py --setsplit=LJ30 --gpuid=1 --model=ResEEGNet --ctr_loss=True --alpha=1e-4  
python -u main.py --setsplit=MI --gpuid=0 --model=ResEEGNet --ctr_loss=True --alpha=1e-4  
python -u main.py --setsplit=SSVEP --gpuid=1 --model=ResEEGNet --ctr_loss=True --alpha=1e-4  
python -u main.py --setsplit=M3CV_Rest --gpuid=0 --model=ResEEGNet --ctr_loss=True --alpha=1e-4  
python -u main.py --setsplit=M3CV_Transient --gpuid=4 --model=ResEEGNet --ctr_loss=True --alpha=1e-4  
python -u main.py --setsplit=M3CV_Steady --gpuid=4 --model=ResEEGNet --ctr_loss=True --alpha=1e-4  
python -u main.py --setsplit=M3CV_P300 --gpuid=4 --model=ResEEGNet --ctr_loss=True --alpha=1e-4  
python -u main.py --setsplit=M3CV_Motor --gpuid=5 --model=ResEEGNet --ctr_loss=True --alpha=1e-4  
python -u main.py --setsplit=M3CV_SSVEP_SA --gpuid=5 --model=ResEEGNet --ctr_loss=True --alpha=1e-4  
python -u main.py --setsplit=SEED --gpuid=2 --model=ResEEGNet --ctr_loss=True --alpha=1e-4  

python -u main.py --setsplit=004 --gpuid=3 --model=FBCNet2   
python -u main.py --setsplit=ERP --gpuid=5 --model=FBCNet2   
python -u main.py --setsplit=LJ30 --gpuid=5 --model=FBCNet2   
python -u main.py --setsplit=MI --gpuid=5 --model=FBCNet2   
python -u main.py --setsplit=SSVEP --gpuid=4 --model=FBCNet2   
python -u main.py --setsplit=M3CV_Rest --gpuid=6 --model=FBCNet2   
python -u main.py --setsplit=M3CV_Transient --gpuid=5 --model=FBCNet2   
python -u main.py --setsplit=M3CV_Steady --gpuid=6 --model=FBCNet2   
python -u main.py --setsplit=M3CV_P300 --gpuid=5 --model=FBCNet2   
python -u main.py --setsplit=M3CV_Motor --gpuid=4 --model=FBCNet2   
python -u main.py --setsplit=M3CV_SSVEP_SA --gpuid=3 --model=FBCNet2   
python -u main.py --setsplit=SEED --gpuid=3 --model=FBCNet2  


tmux
exit
clear



 


