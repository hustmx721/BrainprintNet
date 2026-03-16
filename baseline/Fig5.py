import numpy as np
import sys, os
sys.path.append("/mnt/data1/tyl/UserID/")

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')  # 或者 'Qt5Agg'


class FeatureVisualize(object):
    '''
    Visualize features by TSNE
    '''

    def __init__(self, features, labels):
        '''
        features: (m,n)
        labels: (m,)
        '''
        self.features = features
        self.labels = labels


    def plot_tsne(self, seed=2024, perplexity=30, max_iter=1000):
        ''' Plot TSNE figure. Set save_eps=True if you want to save a .eps file.
        '''
        tsne = TSNE(n_components=2, init='pca', random_state=seed, perplexity=perplexity, max_iter=max_iter)
        features = tsne.fit_transform(self.features)
        x_min, x_max = np.min(features, 0), np.max(features, 0)
        data = (features - x_min) / (x_max - x_min)
        del features
        return data 

"""     
    def plot_tsne(self, seed=2024, save_eps=False, perplexity=30, n_iter=1000):
        ''' Plot TSNE figure. Set save_eps=True if you want to save a .eps file.
        '''
        tsne = TSNE(n_components=2, init='pca', random_state=seed, perplexity=perplexity, n_iter=n_iter)
        features = tsne.fit_transform(self.features)
        x_min, x_max = np.min(features, 0), np.max(features, 0)
        data = (features - x_min) / (x_max - x_min)
        del features
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(self.labels[i]),
                     color=plt.cm.Set1(self.labels[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 8})
        plt.xticks([])
        plt.yticks([])
        plt.title('T-SNE')
        if save_eps:
            plt.savefig('tsne.eps', dpi=600, format='eps')
        plt.show() """


if __name__ == '__main__':
    # P300: 2025/2026, Rest: 2026/2024, Transient: 2024/2028
    split = "M3CV_P300"
    s_seed = 2026
    t_seed = 2026
    
    raw = np.load(f"./tsne/{split}_s{s_seed}_t{t_seed}.npy", allow_pickle=True).item()
    s_feas, s_labels = raw['s_feas'], raw['s_labels']
    t_feas, t_labels = raw['t_feas'], raw['t_labels']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 12), dpi=600, sharex=True, sharey=True)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    # before label smoothing
    p300_remove_list = [0, 2, 3, 5, 6, 8, 11, 12, 13, 14, 15, 18]
    choice1 = np.where(~np.isin(s_labels, p300_remove_list))[0]
    choice2 = np.where(~np.isin(t_labels, p300_remove_list))[0]
    s_fea = s_feas[choice1]
    s_label = s_labels[choice1]
    t_fea = t_feas[choice2]
    t_label = t_labels[choice2]
    print("# p300 paradigm>>>>>>>>>>")
    print("features shape:", s_fea.shape)
    print("labels shape:", s_label.shape)

    s_vis = FeatureVisualize(s_fea, s_label)
    data1 = s_vis.plot_tsne(seed=s_seed)
    t_vis = FeatureVisualize(t_fea, t_label)
    data2 = t_vis.plot_tsne(seed=t_seed)
    

    axes[0].scatter(data1[:, 0], data1[:, 1], c=s_label, cmap='rainbow', alpha=0.6, s=50)
    axes[0].set_title("P300 paradigm before label smoothing", fontsize=10)
    axes[0].set_aspect('equal')

    axes[1].scatter(data2[:, 0], data2[:, 1], c=t_label, cmap='rainbow', alpha=0.6, s=50)
    axes[1].set_title("P300 paradigm after label smoothing", fontsize=10)
    axes[1].set_aspect('equal')

    plt.subplots_adjust(wspace=0.3)  # 可以根据需要调整这个值
    plt.show()