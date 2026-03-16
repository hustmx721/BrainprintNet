python -u within_session.py --setsplit=M3CV_Rest --gpuid=6 --model=FBMSTSNet  &
python -u within_session.py --setsplit=M3CV_Transient --gpuid=5 --model=FBMSTSNet   &
python -u within_session.py --setsplit=M3CV_Steady --gpuid=6 --model=FBMSTSNet   &
python -u within_session.py --setsplit=M3CV_P300 --gpuid=5 --model=FBMSTSNet   &
python -u within_session.py --setsplit=M3CV_Motor --gpuid=4 --model=FBMSTSNet   &
python -u within_session.py --setsplit=M3CV_SSVEP_SA --gpuid=4 --model=FBMSTSNet  