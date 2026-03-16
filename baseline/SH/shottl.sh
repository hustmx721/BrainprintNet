echo "M3CV cross-task & cross-session experiment"
echo "Model MsNet"
echo "positive tranfer"
echo "tl methdod SHOT wo IM loss"

# python -u shot_main.py --cross_task M3CV_Motor M3CV_Rest --gpuid=4 --model=FBMSTSNet   --session_num=21  --tl=SHOT &
# python -u shot_main.py --cross_task M3CV_Motor M3CV_Steady --gpuid=4 --model=FBMSTSNet  --session_num=21  --tl=SHOT &
# python -u shot_main.py --cross_task M3CV_Motor M3CV_P300 --gpuid=4 --model=FBMSTSNet  --session_num=21  --tl=SHOT &
# python -u shot_main.py --cross_task M3CV_Motor M3CV_SSVEP_SA --gpuid=4 --model=FBMSTSNet  --session_num=21  --tl=SHOT &
# python -u shot_main.py --cross_task M3CV_Transient M3CV_Rest --gpuid=6 --model=FBMSTSNet   --session_num=21  --tl=SHOT &
# python -u shot_main.py --cross_task M3CV_Transient M3CV_Steady --gpuid=6 --model=FBMSTSNet  --session_num=21  --tl=SHOT &
# python -u shot_main.py --cross_task M3CV_Transient M3CV_P300 --gpuid=6 --model=FBMSTSNet  --session_num=21  --tl=SHOT &
# python -u shot_main.py --cross_task M3CV_Transient M3CV_SSVEP_SA --gpuid=6 --model=FBMSTSNet  --session_num=21  --tl=SHOT &
# python -u shot_main.py --cross_task M3CV_Rest M3CV_P300 --gpuid=5 --model=FBMSTSNet   --session_num=21  --tl=SHOT &
# python -u shot_main.py --cross_task M3CV_Rest M3CV_SSVEP_SA --gpuid=5 --model=FBMSTSNet  --session_num=21  --tl=SHOT &
# python -u shot_main.py --cross_task M3CV_Steady M3CV_P300 --gpuid=5 --model=FBMSTSNet  --session_num=21  --tl=SHOT &
# python -u shot_main.py --cross_task M3CV_Steady M3CV_SSVEP_SA --gpuid=5 --model=FBMSTSNet  --session_num=21  --tl=SHOT 
 
# python -u shot_main.py --cross_task M3CV_Motor M3CV_Rest --gpuid=0 --model=FBMSTSNet   --session_num=12  --tl=SHOT &
# python -u shot_main.py --cross_task M3CV_Motor M3CV_Steady --gpuid=0 --model=FBMSTSNet  --session_num=12  --tl=SHOT &
# python -u shot_main.py --cross_task M3CV_Motor M3CV_P300 --gpuid=1 --model=FBMSTSNet  --session_num=12  --tl=SHOT &
# python -u shot_main.py --cross_task M3CV_Transient M3CV_Rest --gpuid=1 --model=FBMSTSNet   --session_num=12  --tl=SHOT &
# python -u shot_main.py --cross_task M3CV_Transient M3CV_Steady --gpuid=2 --model=FBMSTSNet  --session_num=12  --tl=SHOT &
# python -u shot_main.py --cross_task M3CV_Transient M3CV_P300 --gpuid=2 --model=FBMSTSNet  --session_num=12  --tl=SHOT 
python -u shot_main.py --cross_task M3CV_Rest M3CV_P300 --gpuid=0 --model=FBMSTSNet   --session_num=12  --tl=SHOT &
python -u shot_main.py --cross_task M3CV_Steady M3CV_P300 --gpuid=0 --model=FBMSTSNet  --session_num=12  --tl=SHOT 