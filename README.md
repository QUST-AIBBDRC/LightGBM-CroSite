##LightGBM-CroSite
Prediction of protein crotonylation sites through LightGBM classifier with multi-feature fusion.

###LightGBM-CroSite uses the following dependencies:
* MATLAB2014a
* python 3.7 
* numpy
* scipy
* scikit-learn
* TensorFlow 
* keras


###Guiding principles:

**The dataset file contains crotonylation sites datasets.

**feature extraction methods:
   BE.m is the implementation of BE.
   EBGW.m is the implementation of EBGW.
   PWAA.m is the implementation of PWAA.
   PsePSSM.m is the implementation of PsePSSM.
   KNN.py is the implementation of KNN.
   
**feature selection methods:
   Elastic_net.py represents the Elastic net.
   Lasso.py represents the Lasso.
   ET.py represents the extra-trees.
   SVD.py represents the singular value decomposition.
   LLE.py represents the locally linear embedding.
   MDS.py represents the multiple dimensional scaling.
   
**SMOTE method： 
   SMOTE.R represents the synthetic minority oversampling technique. 
  
**Classifier:
   NB.py is the implementation of NB.
   AdaBoost.py is the implementation of AdaBoost.
   KNN.py is the implementation of KNN.
   XGBoost.py is the implementation of XGBoost.
   SVM.py is the implementation of SVM.
   RF.py is the implementation of RF.
   LightGBM.py is the implementation of LightGBM.

