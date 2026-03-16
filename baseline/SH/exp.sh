echo "FBMSTSNet experiment"

python -u main.py --setsplit=M3CV_Motor --gpuid=1 --model=FBMSTSNet   --strideFactor=4 &
python -u main.py --setsplit=M3CV_Transient --gpuid=3  --model=FBMSTSNet  --strideFactor=4 &
python -u main.py --setsplit=004 --gpuid=1 --model=FBMSTSNet   --strideFactor=4 &
python -u main.py --setsplit=LJ30 --gpuid=1 --model=FBMSTSNet   --strideFactor=4 &
python -u main.py --setsplit=M3CV_Rest --gpuid=2  --model=FBMSTSNet   --strideFactor=4 &
python -u main.py --setsplit=M3CV_Steady --gpuid=3 --model=FBMSTSNet   --strideFactor=4 &
python -u main.py --setsplit=M3CV_P300 --gpuid=3 --model=FBMSTSNet   --strideFactor=4 &
python -u main.py --setsplit=M3CV_SSVEP_SA --gpuid=3 --model=FBMSTSNet  --strideFactor=4 

# python -u main.py --setsplit=004 --gpuid=2 --model=FBMSTSNet  
# python -u main.py --setsplit=LJ30 --gpuid=0 --model=tryFBMSTSNet_4hz   


