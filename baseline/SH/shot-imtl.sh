python -u shot_main.py --cross_task M3CV_Motor M3CV_Rest --gpuid=0 --model=FBMSTSNet   --session_num=12  --tl=SHOT &
python -u shot_main.py --cross_task M3CV_Motor M3CV_Steady --gpuid=0 --model=FBMSTSNet  --session_num=12  --tl=SHOT &
python -u shot_main.py --cross_task M3CV_Motor M3CV_P300 --gpuid=1 --model=FBMSTSNet  --session_num=12  --tl=SHOT &
python -u shot_main.py --cross_task M3CV_Transient M3CV_Rest --gpuid=1 --model=FBMSTSNet   --session_num=12  --tl=SHOT &
python -u shot_main.py --cross_task M3CV_Transient M3CV_Steady --gpuid=2 --model=FBMSTSNet  --session_num=12  --tl=SHOT &
python -u shot_main.py --cross_task M3CV_Transient M3CV_P300 --gpuid=2 --model=FBMSTSNet  --session_num=12  --tl=SHOT &
python -u shot_main.py --cross_task M3CV_Rest M3CV_P300 --gpuid=0 --model=FBMSTSNet   --session_num=12  --tl=SHOT &
python -u shot_main.py --cross_task M3CV_Steady M3CV_P300 --gpuid=0 --model=FBMSTSNet  --session_num=12  --tl=SHOT 