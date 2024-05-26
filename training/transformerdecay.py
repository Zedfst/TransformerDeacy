#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import keras
import time as tm
from sklearn.metrics import auc,roc_curve,average_precision_score
from sklearn.model_selection import StratifiedKFold
from models import *


# In[ ]:


#data storage path
data_path="../data/"
Codes=np.load(f'{data_path}chf_codes_mimic3.npy')
class_=np.load(f'{data_path}chf_class_mimic3.npy')
Demos=np.load(f'{data_path}chf_demos_mimic3.npy')
Edays=np.load(f'{data_path}chf_edays_mimic3.npy')


# In[ ]:


voc_size=len(list(set(Codes.reshape(-1))))
print(f'Vocabulairy size {voc_size}')


# In[ ]:


#Training******************************************************************************************************

tf.random.set_seed(1234)#Tensorflow random seed for reproducibility
kfold = StratifiedKFold(n_splits=5, shuffle=True,random_state=14)# Cross valiation configuration (please Please document this)
bc=keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0.0)# Loss function (please Please document this)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5, mode='max')# Callback method to prevent overfitting

#Training hyperparameters
batch_size=100
epochs=55
#Metric tables
AUC,AUPRC=[],[]

#Metrics. Default threshold 0.5
threshold=0.5
# precision = tf.keras.metrics.Precision(threshold)
# recall= tf.keras.metrics.Recall(threshold)


for train_index, test_index in kfold.split(Codes,class_):

    #-------------------------------------------------------------------------#
    #Transformer Parameters                                                   #
    # maxlen-> Number of clinical codes per sample                            #
    #voc_size->Number of distinct clinical codes                              #
    #nr_visits-> Number of visits per patient                                 #
    #embed_dim- Embedding dimension                                           #
    #num_heads-> Number of head attention                                     #
    #Learning mode-> with or without decay factor                             #
    #transformer_dropout_rate-> Dropout applied on in the transformer encoder #

    transformer_decay=TransformerDecay(
        maxlen=Codes[train_index].shape[1],
        vocab_size=voc_size+1,
        nr_visits=2,
        embed_dim=80,
        num_heads=1,
        transformer_dropout_rate=0.005,
        learning_mode='with_decay')

    #Compiler
    transformer_decay.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #------------------------------------------------------------#
    #Firts input: Codes
    #Second input: Patient demographics
    #Third input: Elapsed days

    start= tm.perf_counter()
    transformer_decay.fit(
        [Codes[train_index],Demos[train_index],Edays[train_index]],
        class_[train_index],
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_split=0.1)
    finish=tm.perf_counter()
    print(f"Training time {round(finish-start,3)},second(s)")

    #Evaluate the model
    testing_sets=[Codes[test_index],Demos[test_index],Edays[test_index]]
    start= tm.perf_counter()
    hist=transformer_decay.evaluate(testing_sets,
                        class_[test_index],
                        batch_size=batch_size)
    print(f"Testing time {round(finish-start,3)},second(s)")
    

    #Get logits
    Targets_probas = transformer_decay.predict(testing_sets,batch_size=batch_size).ravel()
    fpr,tpr,thresholds=roc_curve(class_[test_index],Targets_probas)
    #AUC score
    print(f'AUC-> {auc(fpr,tpr)}')
    AUC.append(auc(fpr,tpr))

    auprc_ =average_precision_score(class_[test_index],Targets_probas)
    #AUPRC score
    print('AUPRC-> ', auprc_)
    AUPRC.append(auprc_)
    print('\n')

print('\n')
print(f'AUC mean {np.round(np.mean(AUC),3)} AUC std {np.round(np.std(AUC),3)}')#Average of the AUC scores over the 5-fold valiadation
print(f'AUPRC mean {np.round(np.mean(AUPRC),3)} AUC std {np.round(np.std(AUPRC),3)}')#Average of the AUPRC scores over the 5-fold valiadation
print('\n')

