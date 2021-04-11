# encoding=utf-8
"""
    Created on 21:29 2018/11/12 
    @author: Jindong Wang
"""
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=5, mu=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.mu = mu
        self.gamma = gamma
        self.A = None
        self.X = None

    def fit_transform(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        self.X = np.hstack((Xs.T, Xt.T))
        self.X /= np.linalg.norm(self.X, axis=0)
        m, n = self.X.shape
        ns, nt = len(Xs), len(Xt)
        # 求L矩阵(MMD矩阵)
        # (1) 计算论文中的1/n1 与 1/n2，n1表示的是源域的长度，n2表示的是目标域的长度
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        # (2) 计算1/n1^2和1/n2^2与1/n1*n2
        M = e * e.T
        # TODO: 为什么要除以范数？
        M = M / np.linalg.norm(M, 'fro')
        # 求中心矩阵H
        H = np.eye(n) - 1 / n * np.ones((n, n))
        # 求核技巧映射后的矩阵
        K = kernel(self.kernel_type, self.X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.mu * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        # scipy中的eig既可以求解普通特征值也可以求解广义特征值，此处是求解广义特征值，将广义特征问题A*X=lambda*B*X转为C = B^-1*A时求解CX = lambda*X
        # 如果是普通的特征值求解 A*X = lambda*X 可以使用 D,V = numpy.linalg.eig(A) 等同于 D,V = scipy.linalg.eig(A)
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        self.A = V[:, ind[:self.dim]]
        Z = np.dot(self.A.T, K)
        # TODO: 为什么要除以范数？
        Z /= np.linalg.norm(Z, axis=0)
        
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def transform(self, Xt2):
        '''
        Map Xt2 to the latent space created from Xt and Xs
        :param Xs : ns * n_feature, source feature
        :param Xt : nt * n_feature, target feature
        :param Xt2: n_s, n_feature, target feature to be mapped
        :return: Xt2_new, mapped Xt2 with projection created by Xs and Xt
        '''
        
        # Compute kernel with Xt2 as target and X as source
        Xt2 = Xt2.T
        K = kernel(self.kernel_type, X1 = Xt2, X2 = self.X, gamma=self.gamma)
        
        # New target features
        Xt2_new = K @ self.A
        
        return Xt2_new

    
    
if __name__ == '__main__':
    # import pandas as pd
    from sklearn.datasets import make_classification
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    import pickle

    Xs,Ys = make_classification(100,5,random_state=0)
    Xt,Yt = make_classification(10,5,random_state=1)
    
    pickle.dump(Xs,open('dataset/Xs.pkl','wb+'))
    pickle.dump(Ys,open('dataset/Ys.pkl','wb+'))
    pickle.dump(Xt,open('dataset/Xt.pkl','wb+'))
    pickle.dump(Yt,open('dataset/Yt.pkl','wb+'))

    # Xs = pickle.load(open('Xs.pkl','rb'))
    # Ys = pickle.load(open('Ys.pkl','rb'))
    # Xt = pickle.load(open('Xt.pkl','rb'))
    # Yt = pickle.load(open('Yt.pkl','rb'))
    
    # Split target data
    Xt1, Xt2, Yt1, Yt2  = train_test_split(Xt, Yt, train_size=0.5, shuffle=True, random_state=42)
    
    # # Create latent space and evaluate using Xs and Xt1
    tca = TCA(kernel_type='linear', dim=1, mu=1, gamma=1)
    Xs_new,Xt1_new = tca.fit_transform(Xs, Xt1)
    # # Project and evaluate Xt2 existing projection matrix and classifier
    Xt2_new = tca.transform(Xt2)

    # clf = RandomForestClassifier()
    # bse_train_x = np.concatenate([Xs,Xt1],axis=0)
    # bse_train_y = np.concatenate([Ys,Yt1],axis=0)
    # clf.fit(bse_train_x,bse_train_y)
    # pred_tar2 = clf.predict_proba(Xt2)[:,1]
    # print(f'baseline_auc: {roc_auc_score(Yt2,pred_tar2)}')

    # bse_x_new = np.concatenate([Xs_new,Xt1_new],axis=0)
    # clf.fit(bse_x_new,bse_train_y)
    # pred_tar1 = clf.predict_proba(Xt1_new)[:,1]
    # pred_tar2 = clf.predict_proba(Xt2_new)[:,1]
    # auc_tar1 = roc_auc_score(Yt1,pred_tar1)
    # auc_tar2 = roc_auc_score(Yt2,pred_tar2)
    # print(f'AUC of mapped source and target1 data : {auc_tar1:.3f}') 
    # print(f'AUC of mapped target2 data            : {auc_tar2:.3f}') 

forPinv = self.mu * np.identity(n_source + n_target) + np.dot(np.dot(K, L), K)
forPinv[np.isnan(forPinv)] = 0
# 计算(K*L*K+mu*I)^-1 * (K*H*K)
Kc = np.dot(np.dot(np.dot(np.linalg.pinv(forPinv), K), H), K)
Kc[np.isnan(Kc)] = 0
# 对核技巧映射后的矩阵求特征值
w, V = np.linalg.eig(Kc) # 返回特征值与特征向量