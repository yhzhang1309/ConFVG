import pandas as pd
import os
from sklearn import preprocessing
import numpy as np
import copy
import pickle
from to_graph import *

data_route="./Dataset/"
bene=pd.read_csv(data_route+'bene.csv')
claims=pd.read_csv(data_route+'claims.csv')
#Merge two datasets
data_join=pd.merge(claims,bene,on='BeneID',how='left')

#DateDiff=ClaimSEndDt-ClaimStartDt，and drop ClaimEndDt
data_join['DateDiff']=pd.to_datetime(data_join['ClaimEndDt'])-pd.to_datetime(data_join['ClaimStartDt'])
data_join['DateDiff']=data_join['DateDiff'].dt.days
data_join.drop(['ClaimEndDt'],axis=1,inplace=True)

#AdimissionDiff=DischargeDt-AdmissionDt
data_join['AdmissionDiff']=pd.to_datetime(data_join['DischargeDt'])-pd.to_datetime(data_join['AdmissionDt'])
data_join['AdmissionDiff']=data_join['AdmissionDiff'].dt.days

#Normalize the DOD
data_join['DOD']=data_join['DOD'].apply(lambda x: 0 if x=='-1' else 1)

#Normalize all
need_minusone=['ChronicCond_Alzheimer','ChronicCond_Heartfailure',
'ChronicCond_KidneyDisease','ChronicCond_Cancer','ChronicCond_ObstrPulmonary','ChronicCond_Depression',
'ChronicCond_Diabetes','ChronicCond_IschemicHeart','ChronicCond_Osteoporasis','ChronicCond_rheumatoidarthritis','ChronicCond_stroke','Gender','Race']

for i in need_minusone:
    data_join[i]=data_join[i].apply(lambda x:x-1)

need_10000minusone=['ClmAdmitDiagnosisCode','DiagnosisGroupCode']

for i in need_10000minusone:
    data_join[i]=data_join[i].apply(lambda x:x-10000)

#Drop AdmissionDt and DischargeDt
data_join.drop(['AdmissionDt','DischargeDt'],axis=1,inplace=True)

#Divide DOB to year, month and day
def date_to_feature(data,column):
    column_year=column+'_year'
    column_month=column+'_month'
    column_day=column+'_day'
    data[column_year]=data[column].dt.year
    data[column_month]=data[column].dt.month
    data[column_day]=data[column].dt.day
    return data

data_join['DOB']=pd.to_datetime(data_join['DOB'])
data_join=date_to_feature(data_join,'DOB')
data_join.drop(['DOB'],axis=1,inplace=True)
data_join.drop(['DOB_day','PotentialGroupFraud'],axis=1,inplace=True)

#Transform the ClaimStartDt to be processed
data_join['ClaimStartDt']=pd.to_datetime(data_join['ClaimStartDt'])

#Divide the data by day
de_day=1
next_data=[]
start_date=pd.to_datetime('2009-01-01')
while(start_date<pd.to_datetime('2009-12-31')):
    temp=data_join[(data_join['ClaimStartDt']>=start_date)&(data_join['ClaimStartDt']<start_date+pd.Timedelta(days=de_day))]
    next_data.append(temp)
    start_date=start_date+pd.Timedelta(days=de_day)
    
#generate the online dataset
tempnext_data = copy.deepcopy(next_data)
dataset=[]
for i in range(len(next_data)):
    current_data = next_data[i].reset_index(drop=True)
    
    if i < 5:  
        print(f"Graph {i}: nodes={len(current_data)}, max_index={current_data.index.max()}")
    
    next_data_edge = create_edges(current_data, 0)
    next_data_edge = edge_transform(next_data_edge)
    
    if len(next_data_edge) > 0 and i < 5:
        max_edge_index = next_data_edge.max()
        print(f"Graph {i}: max_edge_index={max_edge_index}, nodes={len(current_data)}")
        if max_edge_index >= len(current_data):
            print(f"WARNING: Graph {i} has invalid edge indices!")
    
    next_data_features = pandas_to_numpy(current_data)
    
    dataset.append((next_data_features, next_data_edge))
print('next_data_edge finish')

print(len(dataset))

for i in range(len(dataset)):
    dataset[i] = (dataset[i][0], turn_to_no_direction(dataset[i][1]))
    
print(len(dataset[0][1][0]))

with open(data_route+"datasetonline.dat", "wb") as file:
    pickle.dump(dataset, file)