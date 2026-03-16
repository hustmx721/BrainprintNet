python -u mdd_main.py --cross_task M3CV_Motor M3CV_Rest --gpuid=4 --model=FBMSTSNet   --session_num=12  --tl=MDD &
python -u mdd_main.py --cross_task M3CV_Motor M3CV_Steady --gpuid=4 --model=FBMSTSNet  --session_num=12  --tl=MDD &
python -u mdd_main.py --cross_task M3CV_Motor M3CV_P300 --gpuid=3 --model=FBMSTSNet  --session_num=12  --tl=MDD &
python -u mdd_main.py --cross_task M3CV_Motor M3CV_SSVEP_SA --gpuid=3 --model=FBMSTSNet  --session_num=12  --tl=MDD &
python -u mdd_main.py --cross_task M3CV_Transient M3CV_Rest --gpuid=5 --model=FBMSTSNet   --session_num=12  --tl=MDD &
python -u mdd_main.py --cross_task M3CV_Transient M3CV_Steady --gpuid=5 --model=FBMSTSNet  --session_num=12  --tl=MDD &
python -u mdd_main.py --cross_task M3CV_Transient M3CV_P300 --gpuid=0 --model=FBMSTSNet  --session_num=12  --tl=MDD &
python -u mdd_main.py --cross_task M3CV_Transient M3CV_SSVEP_SA --gpuid=0 --model=FBMSTSNet  --session_num=12  --tl=MDD &
python -u mdd_main.py --cross_task M3CV_Rest M3CV_P300 --gpuid=2 --model=FBMSTSNet   --session_num=12  --tl=MDD &
python -u mdd_main.py --cross_task M3CV_Rest M3CV_SSVEP_SA --gpuid=2 --model=FBMSTSNet  --session_num=12  --tl=MDD &
python -u mdd_main.py --cross_task M3CV_Steady M3CV_P300 --gpuid=1 --model=FBMSTSNet  --session_num=12  --tl=MDD &
python -u mdd_main.py --cross_task M3CV_Steady M3CV_SSVEP_SA --gpuid=1 --model=FBMSTSNet  --session_num=12  --tl=MDD 