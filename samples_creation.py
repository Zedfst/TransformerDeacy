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
import math
from collections import Counter
from utils import samplesConstructor


# In[ ]:


#data storage path
data_path="../data/mimic3/"
admissions=pd.read_csv('data/admissions_processed_mimic3.csv' ,low_memory=False,engine='c')
admissions=admissions.sort_values(by='ADMITTIME')
usecols=['SUBJECT_ID','HADM_ID','ICD9_CODE']#'icd_version','long_title']
diagnoses=pd.read_csv(data_path+'DIAGNOSES_ICD.csv' ,low_memory=False,engine='c')
procedures=pd.read_csv(data_path+'PROCEDURES_ICD.csv' ,usecols=usecols,low_memory=False,engine='c')
usecols=['SUBJECT_ID','HADM_ID','ITEMID']
usecols=['SUBJECT_ID','HADM_ID','GSN']
prescriptions=pd.read_csv(data_path+'PRESCRIPTIONS.csv',usecols=usecols,low_memory=False,engine='c')
icd9_ontologie=pd.read_csv(data_path+'ICD9CM.csv',low_memory=False,engine='c')


# In[ ]:


prescriptions=prescriptions.rename(columns={'GSN':'ICD9_CODE'})
prescriptions=prescriptions.dropna(subset=['ICD9_CODE'])
prescriptions.head(2)


# In[ ]:


no_prescription=int(np.mean(prescriptions.groupby('HADM_ID')['ICD9_CODE'].nunique().values))
print(f' Average number of prescriptions {no_prescription}')


# In[ ]:


hadm_ids=prescriptions.HADM_ID.unique()
# print(hadm_ids)
tempo=pd.DataFrame()
for h in tqdm(hadm_ids):
    df=prescriptions.loc[prescriptions.HADM_ID==h,:]
    df=df.drop_duplicates(subset=['ICD9_CODE'])[:no_prescription]
    tempo=pd.concat([tempo,df],axis=0)


# In[ ]:


prescriptions=tempo
del tempo


# In[ ]:


admissions.head(2)


# In[ ]:


print(f'# patients {admissions.SUBJECT_ID.nunique()}')
print(f'# admissions {admissions.HADM_ID.nunique()}')


# In[ ]:


procedures.head(2)


# In[ ]:


diagnoses.head(2)


# In[ ]:


diagnoses=diagnoses.dropna(subset=['ICD9_CODE'])
procedures=procedures.dropna(subset=['ICD9_CODE'])
prescriptions=prescriptions.dropna(subset=['ICD9_CODE'])

diagnoses['is_diagnosis']=np.ones((len(diagnoses),1))
procedures['is_diagnosis']=np.zeros((len(procedures),1))
prescriptions['is_diagnosis']=np.zeros((len(prescriptions),1))


procedures=procedures.loc[procedures.HADM_ID.isin(diagnoses.HADM_ID.unique()),:]
prescriptions=prescriptions.loc[prescriptions.HADM_ID.isin(diagnoses.HADM_ID.unique()),:]

data=pd.concat([diagnoses,procedures,prescriptions],axis=0)
data=data.dropna(subset=['HADM_ID','ICD9_CODE'])
data.head(3)


# In[ ]:


print(f'# of distinct diagnoses befor strip {data.ICD9_CODE.nunique()}')


# In[ ]:


data=data.rename(columns={'ICD9_CODE':'clinical_code'})
data.head(2)


# In[ ]:


#Encode clinical codes to categorical features
code = data.clinical_code.astype('category').copy()
data['ICD_ENCODED'] = code.cat.codes+1
data.head(2)


# In[ ]:


#Uncomment if you want to save the date for later use
#data.to_csv("df_clinical_codes.csv",index=False)


# In[ ]:


hadmid_icds_cm=data.groupby('HADM_ID')['ICD_ENCODED'].unique().apply(list)
dict_icd_cm={}
for id_,lst in zip(hadmid_icds_cm.index,hadmid_icds_cm.values):
    dict_icd_cm[id_]=np.array(lst)


# In[ ]:


#Dictionary code and binary indication: whether it is a code or not
dict_is_diagnoses={}
tempo_data=data.drop_duplicates(subset=['ICD_ENCODED','is_diagnosis'])
for i,j in zip(tempo_data.ICD_ENCODED.values,tempo_data.is_diagnosis.values):
    dict_is_diagnoses[i]=j
#For padded values
dict_is_diagnoses[0]=0


# In[ ]:


#### Admissions and patients information extraction
usecols_patients=['SUBJECT_ID','GENDER','DOB','DOD']
patients=pd.read_csv(data_path+'PATIENTS.csv',usecols=usecols_patients,parse_dates=['DOB'],low_memory=False,engine='c')
patients.head(2)


# In[ ]:


patients=patients.loc[patients.SUBJECT_ID.isin(admissions.SUBJECT_ID.unique()),:]
admissions=admissions.dropna(subset=['ADMITTIME','ADMISSION_TYPE'])


# In[ ]:


#Age Dictionary
AGES={}
for sbj,date in zip(patients.SUBJECT_ID,patients.DOB):
    admission_age=admissions.loc[admissions.SUBJECT_ID==sbj,:]['ADMITTIME'].values[0]
    admission_age=datetime.strptime(admission_age,'%Y-%m-%d %H:%M:%S')
    days=((admission_age-date).days//366)
    if days<0:
        print(days)
    if days>89:
        days=89
    AGES[sbj]=np.array([days])


# In[ ]:


#Encode patient demographics information
gender_encoded = patients.GENDER.astype('category').copy()
patients['GENDER'] = gender_encoded.cat.codes+1
patients.head(2)


# In[ ]:


patients['AGES']=patients['SUBJECT_ID'].apply(lambda x: AGES[x][0])
patients.head(2)


# In[ ]:


print(f'# of patients {patients.SUBJECT_ID.nunique()}')


# In[ ]:


#Remove a patient under 18
patients=patients.loc[patients.AGES>=18,:]
print(f'# of patients {patients.SUBJECT_ID.nunique()}')


# In[ ]:


admissions=admissions.loc[admissions.SUBJECT_ID.isin(patients.SUBJECT_ID.unique()),:]


# In[ ]:


admission_type_encoded = admissions.ADMISSION_TYPE.astype('category').copy()
admissions['ADMISSION_TYPE_'] = admission_type_encoded.cat.codes+1
admissions.head(3)


# In[ ]:


#Encode insurance type
admission_insurance = admissions.INSURANCE.astype('category').copy()
admissions['INSURANCE'] = admission_insurance.cat.codes+1
admissions.head(3)


# In[ ]:


#Extract the patient's length of stay
icu_data=pd.read_csv(data_path+'ICUSTAYS.csv',low_memory=False,engine='c')
icu_data=icu_data.dropna(subset=['HADM_ID','LOS'])
icu_data.head(2)


# In[ ]:


#Create dictionary for icu data
icu_dict={}
for h,t in zip(icu_data.HADM_ID.values,icu_data.LOS.values):
    icu_dict[int(h)]=t


# In[ ]:


admissions=admissions.loc[admissions.HADM_ID.isin(data.HADM_ID.unique()),:]


# In[ ]:


print(f'# patients {admissions.SUBJECT_ID.nunique()}')


# In[ ]:


number_icd_cm=int(np.max(data.groupby('HADM_ID')['ICD_ENCODED'].nunique().values))
print(f'Maximum # of clinical codes {number_icd_cm}')
nr_historical_visits=2
#Samples construction
Codes,Edays,TypeCodes,Demos,Class=samplesConstructor(admissions,dict_icd_cm,patients,number_icd_cm,nr_historical_visits,icu_dict,dict_is_diagnoses)


# In[ ]:


print(f'Class distributiob {Counter(Class)}')


# In[ ]:


Codes.shape,Class.shape,TypeCodes.shape,Edays.shape,Demos.shape


# In[ ]:


#Save samples
np.save('data/chf_codes_mimic3.npy',Codes)
np.save('data/chf_typecodes_mimic3.npy',TypeCodes)
np.save('data/chf_edays_mimic3.npy',Edays)
np.save('data/chf_demos_mimic3.npy',Demos)
np.save('data/chf_class_mimic3.npy',Class)

