#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras import layers
from sklearn.model_selection import KFold
import sklearn.metrics
import keras.backend as K
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from collections import Counter
import tensorflow as tf
from keras.layers import Dense,Embedding,Flatten,LSTM,Dropout,Bidirectional,MaxPooling1D,Conv1D
from sklearn.metrics import auc,roc_curve,average_precision_score
import time as tm
from models import LSTM_


# In[ ]:


#data storage path
data_path="../data/"
model_name="LR"


# In[ ]:


data_path="../data/"
Codes=np.load(f'{data_path}chf_codes_mimic3.npy')
class_=np.load(f'{data_path}chf_class_mimic3.npy')
Demos=np.load(f'{data_path}chf_demos_mimic3.npy')
Demos[:,4]=np.where(Demos[:,4]<0,0,Demos[:,4])


# In[ ]:


Codes.shape,class_.shape,Demos.shape


# In[ ]:


#Class distibution
print(Counter(class_.reshape(-1)))


# In[ ]:


#Voc_size
voc_size=len(list(set(Codes.reshape(-1))))


# In[ ]:


tf.random.set_seed(1234)
kfold = KFold(n_splits=5, shuffle=True,random_state=14)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5, mode='max')
batch_size=500
epochs=170
AUC,AUPRC=[],[]
for train_index, test_index in kfold.split(Codes,class_):

    lstm=LSTM_(voc_size,50,Codes.shape[1],0.5)
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    start= tm.perf_counter()
    lstm.fit([Codes[train_index],Demos[train_index]],class_[train_index],epochs=epochs,
              batch_size=batch_size,verbose=0,validation_split=0.1)
    finish=tm.perf_counter()
    print(f"Traing time {round(finish-start,3)},second(s)")

    start= tm.perf_counter()
    hist=lstm.evaluate([Codes[test_index],Demos[test_index]],class_[test_index], batch_size=batch_size)
    finish=tm.perf_counter()
    print(f"Testing time {round(finish-start,3)},second(s)")

    Targets_probas = lstm.predict([Codes[test_index],Demos[test_index]]).ravel()
    fpr,tpr,thresholds=roc_curve(class_[test_index],Targets_probas)
    print(f'AUC-> {auc(fpr,tpr)}') 
    AUC.append(auc(fpr,tpr))

    auprc_ =average_precision_score(class_[test_index],Targets_probas)
    print('AUPRC-> ', auprc_)
    AUPRC.append(auprc_)
    print('\n')


print(f'AUC mean {np.round(np.mean(AUC),3)} AUC std {np.round(np.std(AUC),3)}')
print(f'AUPRC mean {np.round(np.mean(AUPRC),3)} AUC std {np.round(np.std(AUPRC),3)}')
print('\n')


# In[ ]:




