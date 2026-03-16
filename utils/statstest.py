# https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
import numpy as np
import pandas as pd
import scipy
import os
from scipy import stats
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel
from scipy.stats import false_discovery_control

def statstest(data1, data2, mode:str='t-test'):
    if mode == 't-test':
        stat, p = ttest_rel(data1, data2)  # t-test
    elif mode == 'wilcoxon':
        stat, p = wilcoxon(data1, data2)  # wilcoxon
    # print(data1.shape)
    # print(data2.shape)
    # print('before control stat={:f}, p={:.32f}'.format(stat, p))
    scipy.stats.false_discovery_control(p)
    confidence_level = None
    # print('stat={:f}, p={:.32f}'.format(stat, p))
    if p > 0.05:
        confidence_level = 0
    elif p < 0.05 and p > 0.01:
        confidence_level = 1
    elif p < 0.01:
        confidence_level = 2
    return p, confidence_level

task = ['AT', 'RS', 'TSS', 'SSS', 'P300', 'ME', 'SSA', 'MI', 'SSVEP']
filelst = ['WPD', 'PSD', 'AR', 'MFCC', 'EEGNet', 'DeepCNN', 'ShallowCNN',
            '1DLSTM', 'Conformer', 'GWNet', 'MSNet', 'IFNet', 'FBCNet', 'FBMSNet']
confidences = np.zeros((len(task), len(filelst)))
probs = np.zeros((len(task), len(filelst)))
root = "/mnt/data1/tyl/UserID/csv/_ACC-t-test/"

for tsk in task:
    tep_dir = os.path.join(root, tsk)
    data1 = np.loadtxt(os.path.join(tep_dir, 'FBMSTSNet' + ".csv"))
    for file in filelst:
        data2 = np.loadtxt(os.path.join(tep_dir, file + ".csv"))
        p, confidence_level = statstest(data1, data2, mode='t-test')
        confidences[task.index(tsk)][filelst.index(file)] = confidence_level
        probs[task.index(tsk)][filelst.index(file)] = p

confidences = pd.DataFrame(confidences, index=task, columns=filelst)
probs = pd.DataFrame(probs, index=task, columns=filelst)

confidences.to_csv(os.path.join(root, 'confidences.csv'))
probs.to_csv(os.path.join(root, 'probs.csv'))

print(confidences)
print(probs)

