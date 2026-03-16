# -------------------------------
# Boxplot for accuracy, f1-score, and eer

import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir("/Users/marx2/BCILab/论文绘图/brainprint绘图/visual/figs")
plt.rcParams['font.family'] = 'Times New Roman'

# 数据准备
# 方法列表和渐变色设置
methods = ["Baseline", "DAN", "DANN", "SHOT", "MDD", "JAN", "MCC", "CDAN", "SHOT-IM"]
colors_acc = plt.cm.Blues(np.linspace(0.4, 1, len(methods)))
colors_f1 = plt.cm.Greens(np.linspace(0.4, 1, len(methods)))
colors_eer = plt.cm.Oranges(np.linspace(0.4, 1, len(methods)))

# Accuracy数据
accuracy_data = [
    [0.4523, 0.5524, 0.5799, 0.5458, 0.4696, 0.5142, 0.5120, 0.5578, 0.4462, 0.5042, 0.5137, 0.5792],
    [0.2780, 0.3100, 0.2301, 0.2142, 0.3184, 0.3133, 0.3527, 0.4105, 0.2522, 0.2242, 0.3278, 0.3083],
    [0.3230, 0.3941, 0.3886, 0.3200, 0.2997, 0.2592, 0.3770, 0.4681, 0.3304, 0.3567, 0.3204, 0.2600],
    [0.4257, 0.5154, 0.4916, 0.4417, 0.4187, 0.4283, 0.5057, 0.5511, 0.3933, 0.4125, 0.4742, 0.4825],
    [0.4417, 0.4905, 0.5177, 0.4625, 0.4314, 0.4367, 0.4883, 0.5411, 0.4288, 0.4367, 0.4763, 0.4658],
    [0.4643, 0.5451, 0.4709, 0.4050, 0.4615, 0.5183, 0.5297, 0.6354, 0.4548, 0.5058, 0.4823, 0.5108],
    [0.4317, 0.5943, 0.4883, 0.3975, 0.4876, 0.4967, 0.6150, 0.7149, 0.4849, 0.5633, 0.5338, 0.5367],
    [0.5263, 0.6927, 0.5933, 0.5250, 0.4381, 0.4475, 0.5817, 0.7143, 0.5237, 0.5058, 0.4957, 0.5042],
    [0.4427, 0.6497, 0.5699, 0.5242, 0.5050, 0.5158, 0.5560, 0.6905, 0.4829, 0.5342, 0.6047, 0.5758]
]

# F1 Score数据
f1_data = [
    [0.4108, 0.5102, 0.5196, 0.4965, 0.4065, 0.4484, 0.4813, 0.5302, 0.4077, 0.4627, 0.4583, 0.5281],  # Baseline
    [0.2770, 0.3045, 0.2493, 0.2495, 0.3330, 0.3578, 0.3320, 0.4144, 0.2683, 0.2630, 0.3289, 0.3568],  # DAN
    [0.3115, 0.3831, 0.4147, 0.3885, 0.3093, 0.2996, 0.3675, 0.4644, 0.3488, 0.4042, 0.3170, 0.3015],  # DANN
    [0.3909, 0.4910, 0.5133, 0.4976, 0.4281, 0.4790, 0.4963, 0.5522, 0.4217, 0.4708, 0.5057, 0.5548],  # SHOTWOIM
    [0.4162, 0.4692, 0.5493, 0.5349, 0.4452, 0.5052, 0.4771, 0.5374, 0.4556, 0.5039, 0.5027, 0.5372],  # MDD
    [0.4806, 0.5601, 0.5493, 0.4995, 0.5320, 0.6432, 0.5434, 0.6590, 0.5249, 0.6266, 0.5476, 0.6256],  # JAN
    [0.4400, 0.6022, 0.5519, 0.4930, 0.5313, 0.6120, 0.6214, 0.7326, 0.5372, 0.6973, 0.5970, 0.6563],  # MCC
    [0.5115, 0.6992, 0.6388, 0.6213, 0.4492, 0.5171, 0.5551, 0.7169, 0.5379, 0.5921, 0.5090, 0.6006],  # CDAN
    [0.4437, 0.6602, 0.6356, 0.6220, 0.5378, 0.6096, 0.5574, 0.7045, 0.5249, 0.6412, 0.6577, 0.6887]   # SHOT
]


# EER数据
eer_data = [
    [0.2882, 0.2356, 0.2207, 0.2390, 0.2785, 0.2557, 0.2568, 0.2327, 0.2920, 0.2610, 0.2562, 0.2215],  # Baseline
    [0.3741, 0.3545, 0.3884, 0.3861, 0.3278, 0.3245, 0.3328, 0.2983, 0.3740, 0.3814, 0.3251, 0.3252],  # DAN
    [0.3492, 0.3071, 0.2915, 0.3206, 0.3425, 0.3612, 0.3187, 0.2657, 0.3201, 0.2981, 0.3263, 0.3606],  # DANN
    [0.2928, 0.2398, 0.2288, 0.2409, 0.2684, 0.2526, 0.2484, 0.2200, 0.2829, 0.2621, 0.2359, 0.2131],  # SHOTWOIM
    [0.2839, 0.2533, 0.2137, 0.2267, 0.2611, 0.2454, 0.2578, 0.2258, 0.2599, 0.2455, 0.2364, 0.2234],  # MDD
    [0.2716, 0.2232, 0.2379, 0.2628, 0.2437, 0.1877, 0.2348, 0.1739, 0.2442, 0.1972, 0.2250, 0.1921],  # JAN
    [0.2894, 0.1960, 0.2267, 0.2693, 0.2291, 0.1999, 0.1878, 0.1304, 0.2264, 0.1596, 0.1968, 0.1773],  # MCC
    [0.2370, 0.1421, 0.1678, 0.1847, 0.2565, 0.2360, 0.2063, 0.1303, 0.2021, 0.2016, 0.2218, 0.1962],  # CDAN
    [0.2828, 0.1660, 0.1802, 0.1849, 0.2184, 0.1932, 0.2198, 0.1438, 0.2285, 0.1782, 0.1551, 0.1505]   # SHOT
]

# 将数据乘以100，便于展示
accuracy_data = np.array(accuracy_data) * 100
f1_data = np.array(f1_data) * 100
eer_data = np.array(eer_data) * 100

# Baseline、Task1、Task2的均值
baseline_avg_acc = np.mean([0.5189] * 9) * 100  # Baseline平均Accuracy
baseline_avg_f1 = np.mean([0.4717] * 9) * 100  # Baseline平均F1
baseline_avg_eer = np.mean([0.2532] * 9) * 100  # Baseline平均EER
task1_avg_acc = np.mean([0.6418] * 9) * 100
task1_avg_f1 = np.mean([0.6020] * 9) * 100
task1_avg_eer = np.mean([0.1885] * 9) * 100
task2_avg_acc = np.mean([0.5714] * 9) * 100
task2_avg_f1 = np.mean([0.5272] * 9) * 100
task2_avg_eer = np.mean([0.2254] * 9) * 100

ACC_AVG = [baseline_avg_acc, task1_avg_acc, task2_avg_acc]
F1_AVG = [baseline_avg_f1, task1_avg_f1, task2_avg_f1]
EER_AVG = [baseline_avg_eer, task1_avg_eer, task2_avg_eer]

    
import matplotlib.pyplot as plt
import numpy as np

def plot_box(data, colors, ylabel, Avg, file_name, show_legend=True):
    avg1, avg2, avg3 = Avg  # baseline, task1, task2的均值
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(7, 7), dpi=400)  # 增大图形尺寸
    
    # 绘制箱线图
    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="black", linewidth=2),
                    boxprops=dict(linewidth=2), whiskerprops=dict(linewidth=2), 
                    flierprops=dict(marker='o', markersize=10, color='red', markerfacecolor='red'))
    
    # 设置坐标轴标签
    ax.set_ylabel(ylabel, fontsize=16)
    
    # 绘制平均值的横线
    if show_legend:
        ax.axhline(avg1, color="crimson", linestyle="-", linewidth=2, label="Baseline average")
        ax.axhline(avg2, color="purple", linestyle='-.', linewidth=2, label="Source task CS average")
        ax.axhline(avg3, color="deeppink", linestyle='--', linewidth=2, label="Source task CS average")
        ax.legend(loc="upper right", fontsize=16)
    else:
        ax.axhline(avg1, color="crimson", linestyle="-", linewidth=2)
        ax.axhline(avg2, color="purple", linestyle='-.', linewidth=2)
        ax.axhline(avg3, color="deeppink", linestyle='--', linewidth=2)
    
    # 设置x轴标签，去掉加粗
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods, rotation=45, fontsize=16)
    ax.tick_params(axis='y', labelsize=14)  # 设置 y 轴刻度字体大小

    # 应用渐变色到箱线图
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # 保存图形
    plt.savefig(file_name, format="png", dpi=400, bbox_inches="tight")
    plt.show()


# 绘制并保存三个箱线图
plot_box(accuracy_data.T, colors_acc, "Accuracy (%)", ACC_AVG, "fig4a_acc.png", show_legend=False)
plot_box(f1_data.T, colors_f1, "F1-Score (%)", F1_AVG, "fig4b_f1.png", show_legend=False)
plot_box(eer_data.T, colors_eer, "EER (%)", EER_AVG, "fig4c_eer.png", show_legend=True)

