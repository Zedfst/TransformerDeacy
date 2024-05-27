#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
import copy
from datetime import datetime,timedelta
import seaborn as sns
from utils import samplesConstructor
from tqdm import tqdm


# In[ ]:


#data storage path
data_path="../data/mimic3/"


# In[ ]:


usecols=['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','ADMISSION_TYPE','INSURANCE','HOSPITAL_EXPIRE_FLAG']
name_admission_file='ADMISSIONS.csv'
admissions=pd.read_csv(data_path+''+name_admission_file,usecols=usecols,low_memory=False,engine='c')
admissions=admissions.sort_values(by="ADMITTIME")


# In[ ]:


admissions.head(2)


# In[ ]:


print(f'Admission types {admissions.ADMISSION_TYPE.unique()}')


# In[ ]:


#Remove newborn
print(f'# of patients {admissions.SUBJECT_ID.nunique()}')
print(f'# of admissions {admissions.HADM_ID.nunique()}')
admissions=admissions.loc[admissions.ADMISSION_TYPE!='NEWBORN',:]
print('Remove newborn')
print(f'# of patients {admissions.SUBJECT_ID.nunique()}')
print(f'# of admissions {admissions.HADM_ID.nunique()}')


# In[ ]:


#Remove deceased patient
admissions=admissions.loc[admissions.HOSPITAL_EXPIRE_FLAG==0,:]
print('Remove deceased patients')
print(f'# of patients {admissions.SUBJECT_ID.nunique()}')
print(f'# of admissions {admissions.HADM_ID.nunique()}')


# In[ ]:


#Compute avergage # of admissions per patients
adm_distrib=admissions.groupby('SUBJECT_ID')['HADM_ID'].count().values
print(f"average admissions per patient {np.mean(adm_distrib)}")


# In[ ]:


admissions.head(2)


# In[ ]:


#Save the admissions data processed
admissions.to_csv('data/admissions_processed_mimic3.csv',index=False)

