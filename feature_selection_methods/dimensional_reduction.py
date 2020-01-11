import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.preprocessing import scale,StandardScaler
from sklearn.preprocessing import normalize,Normalizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso,LassoCV
#from lightning.classification import CDClassifier 
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.linear_model import LassoLarsCV,LassoLars
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import scale,StandardScaler
from sklearn.preprocessing import normalize,Normalizer
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler
from sklearn.decomposition import PCA, NMF, KernelPCA, SparsePCA, FastICA
from sklearn.decomposition import FactorAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS
from sklearn.manifold import SpectralEmbedding 
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import TruncatedSVD


def TSVD(data,n_components=300):
    svd = TruncatedSVD(n_components=n_components)
    new_data=svd.fit_transform(data)  
    return new_data

def LLE(data,n_components=300):
    embedding = LocallyLinearEmbedding(n_components=n_components)
    X_transformed = embedding.fit_transform(data)
    return X_transformed

def mds(data,n_components=300):
    embedding=MDS(n_components=n_components)
    new_data=embedding.fit_transform(data)
    return new_data

	
##using elasticNet to reduce the dimension
def elasticNet(data,label,alpha =np.array([0.09])):
    enetCV = ElasticNetCV(alphas=alpha,l1_ratio=0.1).fit(data,label)
    enet=ElasticNet(alpha=enetCV.alpha_, l1_ratio=0.1)
    enet.fit(data,label)
    mask = enet.coef_ != 0
    new_data = data[:,mask]
    return new_data,mask	
	
##using selectFromExtraTrees to reduce the dimension
def selectFromExtraTrees(data,label):
    clf = ExtraTreesClassifier(n_estimators=10, criterion='gini', max_depth=None, 
                              class_weight=None)#entropy )#entropy
    clf.fit(data,label)
    importance=clf.feature_importances_ 
    model=SelectFromModel(clf,prefit=True)
    new_data = model.transform(data)
    return new_data,importance
 
##using lassodimension to reduce the dimension
def lassodimension(data,label,alpha=np.array([0.01])):#The alpha value range is 0.001, 0.002, 0.005, 0.01, 0.015, 0.02
    lassocv=LassoCV(cv=5, alphas=alpha).fit(data, label)
    x_lasso = lassocv.fit(data,label)#Substituting alpha for dimensionality reduction
    mask = x_lasso.coef_ != 0 
    new_data = data[:,mask] 
    return new_data,mask 


     
