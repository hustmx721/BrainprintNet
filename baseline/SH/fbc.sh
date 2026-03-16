echo "SEED experiment"

# python -u main.py --setsplit=004 --gpuid=0 --model=FBMSNet   &
# python -u main.py --setsplit=LJ30 --gpuid=2 --model=FBMSNet   &
# python -u main.py --setsplit=M3CV_Rest --gpuid=3 --model=FBMSNet   &
# python -u main.py --setsplit=M3CV_Transient --gpuid=0 --model=FBMSNet   &
# python -u main.py --setsplit=M3CV_Steady --gpuid=3 --model=FBMSNet   &
# python -u main.py --setsplit=M3CV_P300 --gpuid=1 --model=FBMSNet   &
# python -u main.py --setsplit=M3CV_Motor --gpuid=2 --model=FBMSNet   &
# python -u main.py --setsplit=M3CV_SSVEP_SA --gpuid=1 --model=FBMSNet   


python -u main.py --setsplit=SEED --gpuid=0 --model=FBCNet  & 
python -u main.py --setsplit=SEED --gpuid=1 --model=FBMSTSNet  &
python -u main.py --setsplit=SEED --gpuid=2 --model=FBMSNet  &
python -u main.py --setsplit=SEED --gpuid=3 --model=MSNet  
# python -u main.py --setsplit=SEED --gpuid=3 --model=MSNet   

