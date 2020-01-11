import scipy.io as sio
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale,StandardScaler
from sklearn.metrics import roc_curve, auc
from dimensional_reduction import LLE
import utils.tools as utils

data_train=pd.read_csv(r'D:\Bioinformatics\JW2\LLE\feature1.csv')
#yeast_data=sio.loadmat('DNN_yeast_six.mat')
data_=np.array(data_train)
data=data_[:,1:]
label=data_[:,0]
shu=scale(data)	
new_X=LLE(shu,n_components=275)
shu=new_X
data_csv = pd.DataFrame(data=shu)
data_csv.to_csv('LLE.csv')
