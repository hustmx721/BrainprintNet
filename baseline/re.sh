# python -u main.py --setsplit=LJ30 --gpuid=0 --model="MCGPnet" &
python -u main.py --setsplit=M3CV_P300 --gpuid=6 --model="MCGPnet" &
# python -u main.py --setsplit=M3CV_Motor --gpuid=5 --model="MCGPnet" &
# python -u main.py --setsplit=M3CV_SSVEP_SA --gpuid=2 --model="MCGPnet" &
# python -u main.py --setsplit=MI --gpuid=3 --model="MCGPnet" & #
# python -u main.py --setsplit=SSVEP --gpuid=4 --model="MCGPnet" & #
python -u main.py --setsplit=M3CV_P300 --gpuid=5 --model="DARNNet" 
# python -u main.py --setsplit=SSVEP --gpuid=6 --model="DARNNet"  #