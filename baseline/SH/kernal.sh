echo "7-15-31-63-127 exp"
python -u main.py --setsplit=M3CV_Rest --gpuid=3 --model=FBMSTSNet &
python -u main.py --setsplit=M3CV_Transient --gpuid=4 --model=FBMSTSNet &
python -u main.py --setsplit=M3CV_Steady --gpuid=5 --model=FBMSTSNet &
python -u main.py --setsplit=M3CV_Motor --gpuid=6 --model=FBMSTSNet