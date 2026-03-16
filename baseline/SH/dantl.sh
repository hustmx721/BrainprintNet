echo "M3CV cross-task experiment"
echo "Model MsNet"
echo "positive and negative tranfer"
echo "tl methdod DAN"

# sleep 600

# python -u dan_main.py --cross_task MI ERP --gpuid=3 --model=FBMSTSNet  --session_num=12 --tl=DAN &8
# python -u dan_main.py --cross_task SSVEP ERP --gpuid=3 --model=FBMSTSNet  --session_num=12 --tl=DAN &
# python -u dan_main.py --cross_task MI ERP --gpuid=3 --model=FBMSTSNet  --session_num=21 --tl=DAN &
# python -u dan_main.py --cross_task SSVEP ERP --gpuid=3 --model=FBMSTSNet  --session_num=21 --tl=DAN

# M3CV模态划分:
# 1. M3CV_Motor(4828-2400)、M3CV_Transient(3590-1791) 70%
# 2. M3CV_Rest(1200-600)、M3CV_Steady(1530-740) 60%
# 3. M3CV_P300(599-299)、M3CV_SSVEP_SA(480-240) 50% 
# 组合:
# [M3CV_Motor, M3CV_Rest] [M3CV_Motor, M3CV_Steady] [M3CV_Motor, M3CV_P300] [M3CV_Motor, M3CV_SSVEP_SA] 
# [M3CV_Transient, M3CV_Rest] [M3CV_Transient, M3CV_Steady] [M3CV_Transient, M3CV_P300] [M3CV_Transient, M3CV_SSVEP_SA] 
# [M3CV_Rest, M3CV_P300] [M3CV_Rest, M3CV_SSVEP_SA] [M3CV_Steady, M3CV_P300] [M3CV_Steady, M3CV_SSVEP_SA] 

# python -u dan_main.py --cross_task M3CV_Motor M3CV_Rest --gpuid=0 --model=FBMSTSNet   --session_num=21  --tl=DAN &
# python -u dan_main.py --cross_task M3CV_Motor M3CV_Steady --gpuid=0 --model=FBMSTSNet  --session_num=21  --tl=DAN &
# python -u dan_main.py --cross_task M3CV_Motor M3CV_P300 --gpuid=0 --model=FBMSTSNet  --session_num=21  --tl=DAN &
# python -u dan_main.py --cross_task M3CV_Motor M3CV_SSVEP_SA --gpuid=0 --model=FBMSTSNet  --session_num=21  --tl=DAN &
# python -u dan_main.py --cross_task M3CV_Transient M3CV_Rest --gpuid=1 --model=FBMSTSNet   --session_num=21  --tl=DAN &
# python -u dan_main.py --cross_task M3CV_Transient M3CV_Steady --gpuid=1 --model=FBMSTSNet  --session_num=21  --tl=DAN &
# python -u dan_main.py --cross_task M3CV_Transient M3CV_P300 --gpuid=1 --model=FBMSTSNet  --session_num=21  --tl=DAN &
# python -u dan_main.py --cross_task M3CV_Transient M3CV_SSVEP_SA --gpuid=1 --model=FBMSTSNet  --session_num=21  --tl=DAN &
# python -u dan_main.py --cross_task M3CV_Rest M3CV_P300 --gpuid=2 --model=FBMSTSNet   --session_num=21  --tl=DAN &
# python -u dan_main.py --cross_task M3CV_Rest M3CV_SSVEP_SA --gpuid=2 --model=FBMSTSNet  --session_num=21  --tl=DAN &
# python -u dan_main.py --cross_task M3CV_Steady M3CV_P300 --gpuid=2 --model=FBMSTSNet  --session_num=21  --tl=DAN &
# python -u dan_main.py --cross_task M3CV_Steady M3CV_SSVEP_SA --gpuid=2 --model=FBMSTSNet  --session_num=21  --tl=DAN 
 

python -u dan_main.py --cross_task M3CV_Motor M3CV_Rest --gpuid=0 --model=FBMSTSNet   --session_num=12  --tl=DAN &
python -u dan_main.py --cross_task M3CV_Motor M3CV_Steady --gpuid=0 --model=FBMSTSNet  --session_num=12  --tl=DAN &
python -u dan_main.py --cross_task M3CV_Motor M3CV_P300 --gpuid=1 --model=FBMSTSNet  --session_num=12  --tl=DAN &
python -u dan_main.py --cross_task M3CV_Transient M3CV_Rest --gpuid=1 --model=FBMSTSNet   --session_num=12  --tl=DAN &
python -u dan_main.py --cross_task M3CV_Transient M3CV_Steady --gpuid=3 --model=FBMSTSNet  --session_num=12  --tl=DAN &
python -u dan_main.py --cross_task M3CV_Transient M3CV_P300 --gpuid=3 --model=FBMSTSNet  --session_num=12  --tl=DAN &
python -u dan_main.py --cross_task M3CV_Rest M3CV_P300 --gpuid=2 --model=FBMSTSNet   --session_num=12  --tl=DAN &
python -u dan_main.py --cross_task M3CV_Steady M3CV_P300 --gpuid=2 --model=FBMSTSNet  --session_num=12  --tl=DAN 