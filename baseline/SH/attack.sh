echo "Adversarial attack on cross-session experiment"
echo "Model FBMSTSNet"
echo "Attack methdod FGSM"


# Attack type: AWP, FGSM
python -u main.py --setsplit=LJ30 --gpuid=1 --model=FBMSTSNet --attack_type=FGSM --attack_coef=1e-3 &
python -u main.py --setsplit=M3CV_Rest --gpuid=1 --model=FBMSTSNet --attack_type=FGSM --attack_coef=1e-3 &
python -u main.py --setsplit=M3CV_Transient --gpuid=1 --model=FBMSTSNet --attack_type=FGSM --attack_coef=1e-3 &
python -u main.py --setsplit=M3CV_Steady --gpuid=3 --model=FBMSTSNet --attack_type=FGSM --attack_coef=1e-3 &
python -u main.py --setsplit=M3CV_P300 --gpuid=3 --model=FBMSTSNet --attack_type=FGSM --attack_coef=1e-3 &
python -u main.py --setsplit=M3CV_Motor --gpuid=5 --model=FBMSTSNet --attack_type=FGSM --attack_coef=1e-3 &
python -u main.py --setsplit=M3CV_SSVEP_SA --gpuid=5 --model=FBMSTSNet --attack_type=FGSM --attack_coef=1e-3 