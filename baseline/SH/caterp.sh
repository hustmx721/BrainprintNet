# python -u main.py --setsplit=CatERP --gpuid=0 --model=1D_LSTM  &
# python -u main.py --setsplit=CatERP --gpuid=0 --model=EEGNet  &
# python -u main.py --setsplit=CatERP --gpuid=1 --model=DeepConvNet  & 
# python -u main.py --setsplit=CatERP --gpuid=1 --model=ShallowConvNet  &
# python -u main.py --setsplit=CatERP --gpuid=2 --model=Conformer  &
# python -u main.py --setsplit=CatERP --gpuid=4 --model=ResEEGNet  &
# python -u main.py --setsplit=CatERP --gpuid=2 --model=MSNet  &
# python -u main.py --setsplit=CatERP --gpuid=4 --model=IFNet   


python -u main.py --setsplit=CatERP --gpuid=1 --model=FBCNet  &
python -u main.py --setsplit=CatERP --gpuid=2 --model=FBMSNet  &
python -u main.py --setsplit=CatERP --gpuid=1 --model=IFNet   

# python -u main.py --setsplit=CatERP --gpuid=0 --model=FBMSTSNet  
