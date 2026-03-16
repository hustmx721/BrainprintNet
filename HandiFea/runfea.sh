# 模糊熵方法计算过于复杂, 效果一般, 不做实验
# 004, LJ30 , "M3CV_Rest", "M3CV_Transient", "M3CV_Steady", "M3CV_P300", "M3CV_Motor", "M3CV_SSVEP_SA"
# python -u ../handifea/fea_main.py  --setsplit=MI --fea_type=wavelet --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=MI --fea_type=PSD --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=MI --fea_type=AR_burg --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=MI --fea_type=MFCC --clf_type=SVM &

# python -u ../handifea/fea_main.py  --setsplit=SSVEP --fea_type=wavelet --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=SSVEP --fea_type=PSD --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=SSVEP --fea_type=AR_burg --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=SSVEP --fea_type=MFCC --clf_type=SVM &

# python -u ../handifea/fea_main.py  --setsplit=ERP --fea_type=wavelet --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=ERP --fea_type=PSD --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=ERP --fea_type=AR_burg --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=ERP --fea_type=MFCC --clf_type=SVM &

# python -u ../handifea/fea_main.py  --setsplit=SEED --fea_type=wavelet --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=SEED --fea_type=PSD --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=SEED --fea_type=AR_burg --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=SEED --fea_type=MFCC --clf_type=SVM 

python -u ../handifea/fea_main.py  --setsplit=CatERP --fea_type=wavelet --clf_type=SVM &
python -u ../handifea/fea_main.py  --setsplit=CatERP --fea_type=PSD --clf_type=SVM &
python -u ../handifea/fea_main.py  --setsplit=CatERP --fea_type=AR_burg --clf_type=SVM &
python -u ../handifea/fea_main.py  --setsplit=CatERP --fea_type=MFCC --clf_type=SVM 
# python -u ../handifea/fea_main.py  --setsplit=M3CV_Rest --fea_type=wavelet --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=M3CV_Rest --fea_type=PSD --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=M3CV_Rest --fea_type=AR_burg --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=M3CV_Rest --fea_type=MFCC --clf_type=SVM &

# python -u ../handifea/fea_main.py  --setsplit=M3CV_Transient --fea_type=wavelet --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=M3CV_Transient --fea_type=PSD --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=M3CV_Transient --fea_type=AR_burg --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=M3CV_Transient --fea_type=MFCC --clf_type=SVM &

# python -u ../handifea/fea_main.py  --setsplit=M3CV_Steady --fea_type=wavelet --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=M3CV_Steady --fea_type=PSD --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=M3CV_Steady --fea_type=AR_burg --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=M3CV_Steady --fea_type=MFCC --clf_type=SVM &

# python -u ../handifea/fea_main.py  --setsplit=M3CV_P300 --fea_type=wavelet --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=M3CV_P300 --fea_type=PSD --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=M3CV_P300 --fea_type=AR_burg --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=M3CV_P300 --fea_type=MFCC --clf_type=SVM &

# python -u ../handifea/fea_main.py  --setsplit=M3CV_Motor --fea_type=wavelet --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=M3CV_Motor --fea_type=PSD --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=M3CV_Motor --fea_type=AR_burg --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=M3CV_Motor --fea_type=MFCC --clf_type=SVM &

# python -u ../handifea/fea_main.py  --setsplit=M3CV_SSVEP_SA --fea_type=wavelet --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=M3CV_SSVEP_SA --fea_type=PSD --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=M3CV_SSVEP_SA --fea_type=AR_burg --clf_type=SVM &
# python -u ../handifea/fea_main.py  --setsplit=M3CV_SSVEP_SA --fea_type=MFCC --clf_type=SVM 