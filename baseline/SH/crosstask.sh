echo "M3CV cross-task experiment"
echo "Model FBMSTSNet"
echo "forward, backward and peer transfer"

# M3CV模态划分:
# 1. M3CV_Motor(4828-2400)、M3CV_Transient(3590-1791) 70%
# 2. M3CV_Rest(1200-600)、M3CV_Steady(1530-740) 60%
# 3. M3CV_P300(599-299)、M3CV_SSVEP_SA(480-240) 50% 
# 组合:
# [M3CV_Motor, M3CV_Rest] [M3CV_Motor, M3CV_Steady] [M3CV_Motor, M3CV_P300] [M3CV_Motor, M3CV_SSVEP_SA] 
# [M3CV_Transient, M3CV_Rest] [M3CV_Transient, M3CV_Steady] [M3CV_Transient, M3CV_P300] [M3CV_Transient, M3CV_SSVEP_SA] 
# [M3CV_Rest, M3CV_P300] [M3CV_Rest, M3CV_SSVEP_SA] [M3CV_Steady, M3CV_P300] [M3CV_Steady, M3CV_SSVEP_SA] 

# session_num=0 表示两个session数据都使用
# python -u cross_task_main.py --cross_task MI ERP --gpuid=3 --model=FBMSTSNet   --session_num=0 &
# python -u cross_task_main.py --cross_task SSVEP ERP --gpuid=4 --model=FBMSTSNet   --session_num=0 &
# python -u cross_task_main.py --cross_task ERP MI --gpuid=5 --model=FBMSTSNet   --session_num=0 &
# python -u cross_task_main.py --cross_task ERP SSVEP --gpuid=6 --model=FBMSTSNet   --session_num=0

# # peer transfer
# python -u cross_task_main.py --cross_task M3CV_Motor M3CV_Transient --gpuid=0 --model=FBMSTSNet   --session_num=0 &
# python -u cross_task_main.py --cross_task M3CV_Transient M3CV_Motor --gpuid=0 --model=FBMSTSNet   --session_num=0 &
# python -u cross_task_main.py --cross_task M3CV_Rest M3CV_Steady --gpuid=1 --model=FBMSTSNet   --session_num=0 &
# python -u cross_task_main.py --cross_task M3CV_Steady M3CV_Rest --gpuid=1 --model=FBMSTSNet   --session_num=0 &
# python -u cross_task_main.py --cross_task M3CV_P300 M3CV_SSVEP_SA --gpuid=2 --model=FBMSTSNet   --session_num=0 &
# python -u cross_task_main.py --cross_task M3CV_SSVEP_SA M3CV_P300 --gpuid=2 --model=FBMSTSNet   --session_num=0 


# Forward transfer
python -u cross_task_main.py --cross_task M3CV_Motor M3CV_Rest --gpuid=0 --model=FBMSTSNet   --session_num=12 &
python -u cross_task_main.py --cross_task M3CV_Motor M3CV_Steady --gpuid=0 --model=FBMSTSNet  --session_num=12 &
python -u cross_task_main.py --cross_task M3CV_Motor M3CV_P300 --gpuid=0 --model=FBMSTSNet  --session_num=12 &
python -u cross_task_main.py --cross_task M3CV_Motor M3CV_SSVEP_SA --gpuid=0 --model=FBMSTSNet  --session_num=12 &
python -u cross_task_main.py --cross_task M3CV_Transient M3CV_Rest --gpuid=1 --model=FBMSTSNet   --session_num=12 &
python -u cross_task_main.py --cross_task M3CV_Transient M3CV_Steady --gpuid=1 --model=FBMSTSNet  --session_num=12 &
python -u cross_task_main.py --cross_task M3CV_Transient M3CV_P300 --gpuid=1 --model=FBMSTSNet  --session_num=12 &
python -u cross_task_main.py --cross_task M3CV_Transient M3CV_SSVEP_SA --gpuid=1 --model=FBMSTSNet  --session_num=12 &
python -u cross_task_main.py --cross_task M3CV_Rest M3CV_P300 --gpuid=2 --model=FBMSTSNet   --session_num=12 &
python -u cross_task_main.py --cross_task M3CV_Rest M3CV_SSVEP_SA --gpuid=2 --model=FBMSTSNet  --session_num=12 &
python -u cross_task_main.py --cross_task M3CV_Steady M3CV_P300 --gpuid=2 --model=FBMSTSNet  --session_num=12 &
python -u cross_task_main.py --cross_task M3CV_Steady M3CV_SSVEP_SA --gpuid=2 --model=FBMSTSNet  --session_num=12 
 

# Backward transfer
# python -u cross_task_main.py --cross_task M3CV_Rest M3CV_Motor --gpuid=4 --model=FBMSTSNet   --session_num=21 &
# python -u cross_task_main.py --cross_task M3CV_Steady M3CV_Motor --gpuid=4 --model=FBMSTSNet  --session_num=21 &
# python -u cross_task_main.py --cross_task M3CV_P300 M3CV_Motor --gpuid=4 --model=FBMSTSNet  --session_num=21 &
# python -u cross_task_main.py --cross_task M3CV_SSVEP_SA M3CV_Motor --gpuid=4 --model=FBMSTSNet  --session_num=21 &
# python -u cross_task_main.py --cross_task M3CV_Rest M3CV_Transient --gpuid=5 --model=FBMSTSNet   --session_num=21 &
# python -u cross_task_main.py --cross_task M3CV_Steady M3CV_Transient --gpuid=5 --model=FBMSTSNet  --session_num=21 &
# python -u cross_task_main.py --cross_task M3CV_P300 M3CV_Transient --gpuid=5 --model=FBMSTSNet  --session_num=21 &
# python -u cross_task_main.py --cross_task M3CV_SSVEP_SA M3CV_Transient --gpuid=5 --model=FBMSTSNet  --session_num=21 &
# python -u cross_task_main.py --cross_task M3CV_P300 M3CV_Rest --gpuid=6 --model=FBMSTSNet   --session_num=21 &
# python -u cross_task_main.py --cross_task M3CV_SSVEP_SA M3CV_Rest --gpuid=6 --model=FBMSTSNet  --session_num=21 &
# python -u cross_task_main.py --cross_task M3CV_P300 M3CV_Steady --gpuid=6 --model=FBMSTSNet  --session_num=21 &
# python -u cross_task_main.py --cross_task M3CV_SSVEP_SA M3CV_Steady --gpuid=6 --model=FBMSTSNet  --session_num=21 





