""" 
instruction:data alignment methods - EA,CA,RA;some for samples(EA,CA),some for sample covariance martix(RA)
Author:hust-marx2
time: 2023/9/18
!lastest:
    1.RA-NotImplemented ,refer to pyriemann(迭代求解方法还没找),老板说用得少不用复现了
    2.CA有点问题,还是直接用现成的
"""
import numpy as np
from functools import reduce
from pyriemann.utils.covariance import covariances
from pyriemann.utils.mean import mean_covariance
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import fractional_matrix_power

# 计算对称正定矩阵(SPD)-X的(-1/2)次方
# 若存在SPD阵(对角矩阵,仅对角线处有元素,且是特征值) B = P^(-1) * A * P,则A^(-1/2) = P * B^(-1/2) * P^(-1)
def func_martix(X):
    # v为特征值、Q为特征向量
    v,Q = np.linalg.eig(X)
    s = np.diag(v**(-0.5))
    # 若出现异常值，则用0代替
    s[np.isnan(s)] = 0
    res = np.dot(Q, np.dot(s,np.linalg.inv(Q)))
    # 取实数部分
    return np.real(res)

# 计算两个矩阵之间的黎曼距离
def Riemann_distance(Q1,Q2):
    mat = np.dot(np.linalg.inv(Q1), Q2)
    # v为特征值
    v,_ = np.linalg.eig(mat)
    RD = np.linalg.norm(np.log(v),"fro") # 计算特征值的F范数
    return RD 


def EA(x, ref=False):
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1])) #(bs,channel,channel)
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5) + (0.00000001) * np.eye(x.shape[1])
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    if ref:
        return XEA, sqrtRefEA
    else:
        return XEA
    
    
# reproduce of EA, CA and RA
class DataAlignment():
    def __init__(self, X, method:str="EA", iter:int=100):
        self.data = X 
        self.method = method
        self.num_trial, self.num_channel, self.num_sample = X.shape
        self.iter = iter # for RA
        
    def __getitem__(self,idx):
        return self.data[idx] # 返回某个trial的数据
    
    # Euclidean Align
    # 主要原理是通过将每个被试者的平均标准协方差矩阵变成单位矩阵，也就是将每个被试者从自身所在域变化到了单位域
    def EA(self):
        EA_data = self.data.copy()
        convar_mat = np.zeros((self.num_trial, self.num_channel, self.num_channel)) # 转换协方差矩阵
        mean_convar_mat = np.zeros((self.num_channel, self.num_channel)) # 平均协方差矩阵
        for idx in range(self.num_trial):
            convar_mat[idx] = np.dot(EA_data[idx,:,:], EA_data[idx,:,:].T)
            # convar_mat[idx] = np.cov(EA_data[idx,:,:])
        # covar_mat_list = [EA_data[i] for i in range(num_trial)]
        # mean_convar_mat = reduce(lambda x,y:x+np.dot(y,y.T), covar_mat_list, mean_convar_mat) / self.num_trail
            mean_convar_mat += convar_mat[idx]
        
        mean_convar_mat = mean_convar_mat / self.num_trial
        R = func_martix(mean_convar_mat) # 变换矩阵
        for idx in range(self.num_trial):
            EA_data[idx,:,:] = np.dot(R, EA_data[idx,:,:]) # 对每个样本作EA
        
        return EA_data
    
    # ! 有点问题，建议直接用现成的
    # 协方差质心对齐（Centroid Alignment，CA）     
    # 使用对数欧几里得均值
    def CA(self):
        CA_data = self.data.copy()
        log_covar_mat_list = np.log(np.array([np.dot(CA_data[i], CA_data[i].T) for i in range(self.num_trial)]))
        mean_convar_mat = np.sum(log_covar_mat_list, axis=0) 
        R = mean_convar_mat /self.num_trial # 平均协方差矩阵
        R_CA = np.exp(R)

        for idx in range(self.num_trial):
            CA_data[idx,:,:] = np.dot(R_CA, CA_data[idx,:,:]) # 对每个样本作CA
        
        return CA_data

    # iter: 迭代计算次数
    # 参考库pyriemann
    def RA(self,iter):
        RA_data = self.data.copy()
        convar_mat = np.array([np.dot(RA_data[i], RA_data[i].T) for i in range(self.num_trial)]) # 协方差矩阵
        G = np.zeros(self.num_channel,self.num_channel) # 0初始化黎曼均值中心

        # 梯度下降法迭代求解?
        for i in range(iter):
            for j in range(self.num_trial):
                all_RD += Riemann_distance(G, convar_mat[j])**2
    

    def __call__(self):
        if self.method == "EA":
            return self.EA()
        elif self.method == "CA":
            return self.CA()
        elif self.method == "RA":
            return self.RA(self.iter)
        else:
            raise NotImplementedError
    
class centroid_align(BaseEstimator, TransformerMixin):
    def __init__(self, center_type='euclid', cov_type='cov'):
        self.center_type = center_type
        self.cov_type = cov_type  # 'cov', 'scm', 'oas'

    def fit(self, X):
        tmp_cov = covariances(X, estimator=self.cov_type) # 协方差矩阵
        center_cov = self._compute_center(tmp_cov, type=self.center_type) # 质心中心协方差矩阵
        if center_cov is None:
            print('mean covriance matrix is none...')
            return None
        self.ref_matrix = fractional_matrix_power(center_cov, -0.5)

        return self

    def transform(self, X):
        num_trial, num_chn, num_point = X.shape[0], X.shape[1], X.shape[2]
        tmp_cov = covariances(X, estimator=self.cov_type)

        cov_new = np.zeros([num_trial, num_chn, num_chn])
        X_new = np.zeros(X.shape)
        for j in range(num_trial):
            trial_cov = np.squeeze(tmp_cov[j, :, :])
            trial_data = np.squeeze(X[j, :, :])
            cov_new[j, :, :] = np.dot(np.dot(self.ref_matrix, trial_cov), self.ref_matrix)
            X_new[j, :, :] = np.dot(self.ref_matrix, trial_data)
        return cov_new, X_new

    def fit_transform(self, X, **kwargs):
        tmp_cov = covariances(X, estimator=self.cov_type)
        center_cov = self._compute_center(tmp_cov, type=self.center_type)
        if center_cov is None:
            print('mean covriance matrix is none...')
            return None
        ref_matrix = fractional_matrix_power(center_cov, -0.5)

        num_trial, num_chn, num_point = X.shape[0], X.shape[1], X.shape[2]
        cov_new = np.zeros([num_trial, num_chn, num_chn])
        X_new = np.zeros(X.shape)
        for j in range(num_trial):
            trial_cov = np.squeeze(tmp_cov[j, :, :])
            trial_data = np.squeeze(X[j, :, :])
            cov_new[j, :, :] = np.dot(np.dot(ref_matrix, trial_cov), ref_matrix)
            X_new[j, :, :] = np.dot(ref_matrix, trial_data)
        return cov_new, X_new

    def _compute_center(self, tmp_cov, type='logeuclid'):
        center_cov = None
        if type == 'riemann':
            center_cov = mean_covariance(tmp_cov, metric='riemann')
        elif type == 'logeuclid':
            center_cov = mean_covariance(tmp_cov, metric='logeuclid')
        elif type == 'euclid': # 欧几里得中心
            center_cov = np.mean(tmp_cov, axis=0)
        else:
            print("unsupport center...")
        return center_cov
""" 
if __name__ == '__main__':
    X = np.random.rand(144, 22, 750)  # 144*22*750
    print(X.shape)

    # way 1
    ca = centroid_align(center_type='euclid', cov_type='lwf')
    ca.fit(X)
    cov_new, X_new = ca.transform(X)
    print(cov_new.shape, X_new.shape)

    # way 2
    ca = centroid_align(center_type='euclid', cov_type='lwf')
    cov_new, X_new = ca.fit_transform(X)
    print(cov_new.shape, X_new.shape) 
"""