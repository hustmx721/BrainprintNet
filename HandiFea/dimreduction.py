""" 
instruction: dimensionality reduction methods
Author:hust-marx2
time: 2024/1/23
lastest:
"""

from sklearn.decomposition import PCA



def PCA_feature(feature):
    pca_model = PCA(n_components=0.99) 
    feature = feature.reshape((feature.shape[0], -1))
    feature_new = pca_model.fit_transform(feature)
    return feature_new