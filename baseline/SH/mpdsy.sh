python -u cdan_main.py --cross_task M3CV_Transient M3CV_SSVEP_SA --gpuid=4 --model=FBMSTSNet  --session_num=12 --tl=CDAN &
python -u cdan_main.py --cross_task M3CV_Rest M3CV_SSVEP_SA --gpuid=4 --model=FBMSTSNet  --session_num=12 --tl=CDAN &
python -u cdan_main.py --cross_task M3CV_Steady M3CV_SSVEP_SA --gpuid=4 --model=FBMSTSNet  --session_num=12 --tl=CDAN &
python -u cdan_main.py --cross_task M3CV_Motor M3CV_SSVEP_SA --gpuid=4 --model=FBMSTSNet  --session_num=12 --tl=CDAN


# python -u dan_main.py --cross_task M3CV_Motor M3CV_P300 --gpuid=6 --model=FBMSTSNet  --session_num=12 --tl=DAN &
# python -u dan_main.py --cross_task M3CV_Motor M3CV_SSVEP_SA --gpuid=6 --model=FBMSTSNet  --session_num=12 --tl=DAN &
# python -u mdd_main.py --cross_task M3CV_Motor M3CV_P300 --gpuid=6 --model=FBMSTSNet  --session_num=12 --tl=MDD &
# python -u mdd_main.py --cross_task M3CV_Rest M3CV_P300 --gpuid=5 --model=FBMSTSNet  --session_num=12 --tl=MDD &
# python -u mdd_main.py --cross_task M3CV_Steady M3CV_P300 --gpuid=5 --model=FBMSTSNet  --session_num=12 --tl=MDD &
# python -u mdd_main.py --cross_task M3CV_Transient M3CV_P300 --gpuid=5 --model=FBMSTSNet  --session_num=12 --tl=MDD 

