echo "12hz exp"
python -u main.py --setsplit=M3CV_Rest --gpuid=6 --model=FBMSTSNet  &
python -u main.py --setsplit=M3CV_Transient --gpuid=6 --model=FBMSTSNet   &
python -u main.py --setsplit=M3CV_Steady --gpuid=6 --model=FBMSTSNet   &
python -u main.py --setsplit=M3CV_P300 --gpuid=6 --model=FBMSTSNet   
python -u main.py --setsplit=M3CV_Motor --gpuid=6 --model=FBMSTSNet   &
python -u main.py --setsplit=M3CV_SSVEP_SA --gpuid=6 --model=FBMSTSNet   