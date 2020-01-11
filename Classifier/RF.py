import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import scale,StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils.tools as utils
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
data_train=pd.read_csv(r'SMOTE1_NET_0.03.csv',header=0)
data_=np.array(data_train)
data=data_[:,2:]
label=data_[:,1]
def get_shuffle(data,label):    
    #shuffle data
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    data = data[index]
    label = label[index]
    return data,label  
X_=scale(data)
X,y=get_shuffle(X_,label)
sepscores = []
y_score=np.ones((1,2))*0.5
y_class=np.ones((1,1))*0.5       
#cv_clf = AdaBoostClassifier(n_estimators=50,learning_rate=0.1)

loo = LeaveOneOut()
for train, test in loo.split(X):
    cv_clf = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=10, 
                                min_samples_split=2, min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, max_features='auto', 
                                max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                min_impurity_split=None, bootstrap=True, 
                                oob_score=False, n_jobs=1, random_state=None, verbose=0, 
                                warm_start=False, class_weight=None)
    X_train=X[train]
    y_train=y[train] 
    X_test=X[test]
    y_test=y[test]
    y_sparse=utils.to_categorical(y)
    y_train_sparse=utils.to_categorical(y_train)
    y_test_sparse=utils.to_categorical(y_test)
    hist=cv_clf.fit(X_train, y_train)
    y_predict_score=cv_clf.predict_proba(X_test) 
    y_predict_class= utils.categorical_probas_to_classes(y_predict_score)
    y_score=np.vstack((y_score,y_predict_score))
    y_class=np.vstack((y_class,y_predict_class))
    cv_clf=[]
y_class=y_class[1:]
y_score=y_score[1:]
fpr, tpr, _ = roc_curve(y_sparse[:,0], y_score[:,0])
roc_auc = auc(fpr, tpr)
acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class, y)
result=[acc,precision,npv,sensitivity,specificity,mcc,roc_auc]
row=y_score.shape[0]
#column=data.shape[1]
y_sparse=utils.to_categorical(y)
yscore_sum = pd.DataFrame(data=y_score)
yscore_sum.to_csv('y_score_RF.csv')
ytest_sum = pd.DataFrame(data=y_sparse)
ytest_sum.to_csv('y_test_RF.csv')
fpr, tpr, _ = roc_curve(y_sparse[:,0], y_score[:,0])
auc_score=result[6]
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='SVM ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('RF.csv')
