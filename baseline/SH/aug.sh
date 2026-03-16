echo "Data augmentation cross-session experiment"
echo "Model FBMSTSNet"
echo "aug methdod channel_mixure"


# Data aug type: channel_mixup trial_mixup channel_reverse channel_noise(pink/gaussian) use_DWTA channel_mixure
python -u main.py --setsplit=LJ30 --gpuid=3 --model=FBMSTSNet --aug_type=channel_mixure &
python -u main.py --setsplit=M3CV_Rest --gpuid=4 --model=FBMSTSNet --aug_type=channel_mixure &
python -u main.py --setsplit=M3CV_Transient --gpuid=4 --model=FBMSTSNet --aug_type=channel_mixure &
python -u main.py --setsplit=M3CV_Steady --gpuid=4 --model=FBMSTSNet --aug_type=channel_mixure &
python -u main.py --setsplit=M3CV_P300 --gpuid=3 --model=FBMSTSNet --aug_type=channel_mixure &
python -u main.py --setsplit=M3CV_Motor --gpuid=3 --model=FBMSTSNet --aug_type=channel_mixure &
python -u main.py --setsplit=M3CV_SSVEP_SA --gpuid=3 --model=FBMSTSNet --aug_type=channel_mixure 