echo "sf5 exp"
python -u main.py --setsplit=M3CV_Rest --gpuid=0 --model=FBMSTSNet --strideFactor=5 &
python -u main.py --setsplit=M3CV_Transient --gpuid=1 --model=FBMSTSNet --strideFactor=5 &
python -u main.py --setsplit=M3CV_Steady --gpuid=5 --model=FBMSTSNet --strideFactor=5 &
python -u main.py --setsplit=M3CV_Motor --gpuid=2 --model=FBMSTSNet --strideFactor=5