echo "M3CV cross-task experiment"
echo "Model FBMSTSNet"
echo "positive and negative tranfer"
echo "tl methdod MCC"


# python -u mcc_main.py --cross_task MI ERP --gpuid=0 --model=FBMSTSNet  --session_num=12 --tl=MCC &
# python -u mcc_main.py --cross_task SSVEP ERP --gpuid=5 --model=FBMSTSNet  --session_num=12 --tl=MCC &
# python -u mcc_main.py --cross_task MI ERP --gpuid=6 --model=FBMSTSNet  --session_num=21 --tl=MCC &
# python -u mcc_main.py --cross_task SSVEP ERP --gpuid=1 --model=FBMSTSNet  --session_num=21 --tl=MCC 

# python -u mcc_main.py --cross_task M3CV_Motor M3CV_Rest --gpuid=4 --model=FBMSTSNet   --session_num=21  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Motor M3CV_Steady --gpuid=4 --model=FBMSTSNet  --session_num=21  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Motor M3CV_P300 --gpuid=4 --model=FBMSTSNet  --session_num=21  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Motor M3CV_SSVEP_SA --gpuid=4 --model=FBMSTSNet  --session_num=21  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Transient M3CV_Rest --gpuid=6 --model=FBMSTSNet   --session_num=21  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Transient M3CV_Steady --gpuid=6 --model=FBMSTSNet  --session_num=21  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Transient M3CV_P300 --gpuid=6 --model=FBMSTSNet  --session_num=21  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Transient M3CV_SSVEP_SA --gpuid=5 --model=FBMSTSNet  --session_num=21  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Rest M3CV_P300 --gpuid=5 --model=FBMSTSNet   --session_num=21  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Rest M3CV_SSVEP_SA --gpuid=5 --model=FBMSTSNet  --session_num=21  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Steady M3CV_P300 --gpuid=4 --model=FBMSTSNet  --session_num=21  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Steady M3CV_SSVEP_SA --gpuid=4 --model=FBMSTSNet  --session_num=21  --tl=MCC 
 
# python -u mcc_main.py --cross_task M3CV_Motor M3CV_Rest --gpuid=4 --model=FBMSTSNet   --session_num=12  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Motor M3CV_Steady --gpuid=4 --model=FBMSTSNet  --session_num=12  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Motor M3CV_P300 --gpuid=3 --model=FBMSTSNet  --session_num=12  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Motor M3CV_SSVEP_SA --gpuid=3 --model=FBMSTSNet  --session_num=12  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Transient M3CV_Rest --gpuid=5 --model=FBMSTSNet   --session_num=12  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Transient M3CV_Steady --gpuid=5 --model=FBMSTSNet  --session_num=12  --tl=MCC 
# python -u mcc_main.py --cross_task M3CV_Transient M3CV_P300 --gpuid=6 --model=FBMSTSNet  --session_num=12  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Transient M3CV_SSVEP_SA --gpuid=6 --model=FBMSTSNet  --session_num=12  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Rest M3CV_P300 --gpuid=2 --model=FBMSTSNet   --session_num=12  --tl=MCC &
# python -u mcc_main.py --cross_task M3CV_Rest M3CV_SSVEP_SA --gpuid=2 --model=FBMSTSNet  --session_num=12  --tl=MCC 
python -u mcc_main.py --cross_task M3CV_Steady M3CV_P300 --gpuid=6 --model=FBMSTSNet  --session_num=12  --tl=MCC &
python -u mcc_main.py --cross_task M3CV_Steady M3CV_SSVEP_SA --gpuid=6 --model=FBMSTSNet  --session_num=12  --tl=MCC 