echo "Label smoothing cross-session experiment"
echo "Model FBMSTSNet"

# python -u main.py --setsplit=LJ30 --gpuid=4 --model=FBMSTSNet  &
# python -u main.py --setsplit=M3CV_Rest --gpuid=4 --model=FBMSTSNet  &
# python -u main.py --setsplit=M3CV_Transient --gpuid=6 --model=FBMSTSNet  &
# python -u main.py --setsplit=M3CV_Steady --gpuid=5 --model=FBMSTSNet  &
# python -u main.py --setsplit=M3CV_P300 --gpuid=5 --model=FBMSTSNet  &
# python -u main.py --setsplit=M3CV_Motor --gpuid=4 --model=FBMSTSNet  &
# python -u main.py --setsplit=M3CV_SSVEP_SA --gpuid=6 --model=FBMSTSNet  

python -u main.py --setsplit=MI --gpuid=5 --model=FBMSTSNet  &
python -u main.py --setsplit=SSVEP --gpuid=6 --model=FBMSTSNet  