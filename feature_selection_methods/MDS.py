import scipy.io as sio
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.ensemble import (RandomForestClassifier,ExtraTreesClassifier)
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing import scale
import utils.tools as utils
from sklearn.metrics import roc_curve, auc
from dimensional_reduction import mds
import pandas as pd
import matplotlib.pyplot as plt

data_train=pd.read_csv(r'D:\Bioinformatics\JW2\MDS\feature1.csv')
#yeast_data=sio.loadmat('DNN_yeast_six.mat')
data_=np.array(data_train)
data=data_[:,1:]
label=data_[:,0]
shu=scale(data)
data_2=mds(shu,n_components=275)
shu=data_2
data_csv = pd.DataFrame(data=shu)
data_csv.to_csv('MDS.csv')

