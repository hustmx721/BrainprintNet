echo "M3CV cross-task & cross-session experiment"
echo "Model MsNet"
echo "positive and negative tranfer"
echo "tl methdod SHOT +MCC"


python -u shotmcc.py --cross_task M3CV_Motor M3CV_Rest --gpuid=2 --model=MSNet   --session_num=21  --tl=SHOTMCC &
python -u shotmcc.py --cross_task M3CV_Motor M3CV_Steady --gpuid=2 --model=MSNet  --session_num=21  --tl=SHOTMCC &
python -u shotmcc.py --cross_task M3CV_Motor M3CV_P300 --gpuid=3 --model=MSNet  --session_num=21  --tl=SHOTMCC &
python -u shotmcc.py --cross_task M3CV_Motor M3CV_SSVEP_SA --gpuid=3 --model=MSNet  --session_num=21  --tl=SHOTMCC &
python -u shotmcc.py --cross_task M3CV_Transient M3CV_Rest --gpuid=6 --model=MSNet   --session_num=21  --tl=SHOTMCC &
python -u shotmcc.py --cross_task M3CV_Transient M3CV_Steady --gpuid=6 --model=MSNet  --session_num=21  --tl=SHOTMCC &
python -u shotmcc.py --cross_task M3CV_Transient M3CV_P300 --gpuid=6 --model=MSNet  --session_num=21  --tl=SHOTMCC &
python -u shotmcc.py --cross_task M3CV_Transient M3CV_SSVEP_SA --gpuid=6 --model=MSNet  --session_num=21  --tl=SHOTMCC &
python -u shotmcc.py --cross_task M3CV_Rest M3CV_P300 --gpuid=5 --model=MSNet   --session_num=21  --tl=SHOTMCC &
python -u shotmcc.py --cross_task M3CV_Rest M3CV_SSVEP_SA --gpuid=5 --model=MSNet  --session_num=21  --tl=SHOTMCC &
python -u shotmcc.py --cross_task M3CV_Steady M3CV_P300 --gpuid=4 --model=MSNet  --session_num=21  --tl=SHOTMCC &
python -u shotmcc.py --cross_task M3CV_Steady M3CV_SSVEP_SA --gpuid=4 --model=MSNet  --session_num=21  --tl=SHOTMCC 
 
# python -u shotmcc.py --cross_task M3CV_Motor M3CV_Rest --gpuid=0 --model=MSNet   --session_num=12  --tl=SHOTMCC &
# python -u shotmcc.py --cross_task M3CV_Motor M3CV_Steady --gpuid=0 --model=MSNet  --session_num=12  --tl=SHOTMCC &
# python -u shotmcc.py --cross_task M3CV_Motor M3CV_P300 --gpuid=0 --model=MSNet  --session_num=12  --tl=SHOTMCC &
# python -u shotmcc.py --cross_task M3CV_Motor M3CV_SSVEP_SA --gpuid=0 --model=MSNet  --session_num=12  --tl=SHOTMCC &
# python -u shotmcc.py --cross_task M3CV_Transient M3CV_Rest --gpuid=1 --model=MSNet   --session_num=12  --tl=SHOTMCC &
# python -u shotmcc.py --cross_task M3CV_Transient M3CV_Steady --gpuid=1 --model=MSNet  --session_num=12  --tl=SHOTMCC &
# python -u shotmcc.py --cross_task M3CV_Transient M3CV_P300 --gpuid=1 --model=MSNet  --session_num=12  --tl=SHOTMCC &
# python -u shotmcc.py --cross_task M3CV_Transient M3CV_SSVEP_SA --gpuid=1 --model=MSNet  --session_num=12  --tl=SHOTMCC &
# python -u shotmcc.py --cross_task M3CV_Rest M3CV_P300 --gpuid=4 --model=MSNet   --session_num=12  --tl=SHOTMCC &
# python -u shotmcc.py --cross_task M3CV_Rest M3CV_SSVEP_SA --gpuid=4 --model=MSNet  --session_num=12  --tl=SHOTMCC &
# python -u shotmcc.py --cross_task M3CV_Steady M3CV_P300 --gpuid=5 --model=MSNet  --session_num=12  --tl=SHOTMCC &
# python -u shotmcc.py --cross_task M3CV_Steady M3CV_SSVEP_SA --gpuid=5 --model=MSNet  --session_num=12  --tl=SHOTMCC 