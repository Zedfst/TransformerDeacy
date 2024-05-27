#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras import layers
from sklearn.model_selection import KFold
import sklearn.metrics
import tensorflow as tf
import pandas as pd
import numpy as np
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense,Embedding,Flatten,LSTM,Dropout,Bidirectional,MaxPooling1D,Conv1D
from sklearn.metrics import auc,roc_curve,average_precision_score
import keras
from sklearn.linear_model import LogisticRegression
import time as tm


# In[ ]:


data_path="../data/"
Codes=np.load(f'{data_path}chf_codes_mimic3.npy')
class_=np.load(f'{data_path}chf_class_mimic3.npy')
Demos=np.load(f'{data_path}chf_demos_mimic3.npy')
Demos[:,4]=np.where(Demos[:,4]<0,0,Demos[:,4])


# In[ ]:


Codes.shape,class_.shape,Demos.shape


# In[ ]:


Demos=Demos.astype(np.float32)
class_=class_.reshape(-1)


# In[ ]:


Demos[:,0]=Demos[:,0]/np.max(Demos[:,0])


# In[ ]:


dummies_codes=np.zeros_like(np.random.rand(Codes.shape[0],np.max(Codes)))
for i1,code in enumerate(Codes):
    for i2,cd in enumerate(code):
        if cd!=0:
            dummies_codes[i1][cd-1]=1


# In[ ]:


dummies_codes=np.concatenate([dummies_codes,Demos],1)


# In[ ]:


print(f"Dummy codes + patient's demographis shape {dummies_codes.shape}")
print(f'States shape {class_.shape}')


# In[ ]:


kfold = KFold(n_splits=5, shuffle=True,random_state=14)
max_iter=10
AUC,AUPRC=[],[]
for train_index, test_index in kfold.split(dummies_codes,class_):

    lr=LogisticRegression(random_state=0,
                          verbose=0,
                          max_iter=max_iter)
    start= tm.perf_counter()
    classification = lr.fit(dummies_codes[train_index], class_[train_index])
    finish=tm.perf_counter()
    print(f"Training time {round(finish-start,3)},second(s)")
    
    
    Targets_probas=classification.predict_proba(dummies_codes[test_index])
    fpr,tpr,thresholds=roc_curve(class_[test_index],Targets_probas[:,1])
    print(f'AUC-> {auc(fpr,tpr)}') 
    AUC.append(auc(fpr,tpr))
    
    auprc_ =average_precision_score(class_[test_index],Targets_probas[:,1])
    print('AUPRC-> ', auprc_)
    AUPRC.append(auprc_)
    print('\n')
    
print(f'AUC mean {np.round(np.mean(AUC),3)} AUC std {np.round(np.std(AUC),3)}')
print(f'AUPRC mean {np.round(np.mean(AUPRC),3)} AUC std {np.round(np.std(AUPRC),3)}')


# In[ ]:




