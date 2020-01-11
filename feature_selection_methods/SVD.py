import numpy as np
import pandas as pd
from sklearn.preprocessing import scale,StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
import scipy.io as sio
from L1_Matine import  TSVD

#data_train=pd.read_csv(r'D:\Bioinformatics\JW2\SVD\feature.csv')
#data_=np.array(data_train)
data_train=sio.loadmat(r'D:\Bioinformatics\JW2\SVD\feature1.mat')
data_=data_train.get('x')
data=data_[:,2:]
shu=scale(data)
X=shu
svd = TruncatedSVD(n_components=217, n_iter=10, random_state=42)
hist=svd.fit(X)  
new_data=svd.transform(X)
data_csv = pd.DataFrame(data=new_data)
data_csv.to_csv('SVD.csv')