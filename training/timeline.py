#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras import layers
from sklearn.model_selection import KFold
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from collections import Counter
from keras.models import Sequential
from sklearn.metrics import auc,roc_curve,average_precision_score# from utils import getSamples
from tqdm import tqdm
import copy
import time as tm
from models import Timeline


# In[ ]:


#data storage path
data_path="../data/"


# In[ ]:


#Load data
Codes=np.load(f'{data_path}chf_codes_mimic3.npy')
class_=np.load(f'{data_path}chf_class_mimic3.npy')
Demos=np.load(f'{data_path}chf_demos_mimic3.npy')
Demos[:,4]=np.where(Demos[:,4]<0,0,Demos[:,4])
Edays=np.load(f'{data_path}chf_edays_mimic3.npy')
Edays=np.flip(Edays,axis=1)
# TypeCodes=np.load(f'{data_path}typecodes_mimic4.npy')
print(Codes.shape,Edays.shape,Demos.shape,class_.shape)


# In[ ]:


print(Codes.shape,class_.shape,Demos.shape,Edays.shape)


# In[ ]:


distribution=Counter(class_.reshape(-1))
distribution


# In[ ]:


#Voc_size for the embedding layer
voc_size=len(list(set(Codes.reshape(-1))))


# In[ ]:


tf.random.set_seed(1234)
kfold = KFold(n_splits=5, shuffle=True,random_state=14)
bc=keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0.0)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=5, mode='max')
batch_size=500
epochs=2

AUC,AUPRC=[],[]
for train_index, test_index in kfold.split(Codes,class_):

    timeline=Timeline(voc_size,50,
              Codes.shape[1])

    timeline.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    start= tm.perf_counter()
    timeline.fit([Codes[train_index],Edays[train_index],Demos[train_index]],class_[train_index],
              epochs=epochs,
              batch_size=batch_size,
              #callbacks=[callback],
              verbose=0,validation_split=0.1)
    finish=tm.perf_counter()
    print(f"Training time {round(finish-start,3)},second(s)")
    
    start= tm.perf_counter()
    hist=timeline.evaluate([Codes[test_index],Edays[test_index],Demos[test_index]],class_[test_index], batch_size=100)
    finish=tm.perf_counter()
    print(f"Testing time {round(finish-start,3)},second(s)")
    
    Targets_probas = timeline.predict([Codes[test_index],Edays[test_index],Demos[test_index]]).ravel()
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

