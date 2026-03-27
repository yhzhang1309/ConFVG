import pandas as pd
from sklearn import preprocessing
import numpy as np

def create_edges(data,large=0):
    edges=[]
    datagroup=data.groupby(['BeneID'])
    for i in datagroup:
        if(len(i[1])>1):
            index=i[1].index
            for j in range(len(index)):
                indexa=index[j]
                for k in range(j+1,len(index)):
                    indexb=index[k]
                    if((large==0) or i[1].iloc[j]['ClaimStartDt']-i[1].iloc[k]['ClaimStartDt']<pd.Timedelta(days=5)):
                        if(indexa<indexb):
                            edges.append((indexa,indexb))
                        else:
                            edges.append((indexb,indexa))


    providergroup=data.groupby(['Provider'])
    for i in providergroup:
        if(len(i[1])>1):
            index=i[1].index
            for j in range(len(index)):
                indexa=index[j]
                for k in range(j+1,len(index)):
                    indexb=index[k]
                    if((large==0)or i[1].iloc[j]['ClaimStartDt']-i[1].iloc[k]['ClaimStartDt']<pd.Timedelta(days=5)):
                        if(indexa<indexb):
                            edges.append((indexa,indexb))
                        else:
                            edges.append((indexb,indexa))
    return list(set(edges))

def edge_transform(edges):
    edges=np.array(edges)
    edges=edges.T.reshape(2,-1)
    return edges

def pandas_to_numpy(data):
    data=data.drop(['BeneID','Provider','ClaimStartDt','ClaimID'],axis=1)
    data=data.values
    data=preprocessing.MinMaxScaler().fit_transform(data)
    return data

def turn_to_no_direction(edge):
    edge_begin=edge[0]
    edge_end=edge[1]
    begin_new_begin=np.concatenate((edge_begin,edge_end))
    begin_new_end=np.concatenate((edge_end,edge_begin))
    return np.array([begin_new_begin,begin_new_end])
