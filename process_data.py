import pandas as pd
import numpy as np
import torch
import torch.utils.data as data_utils
import os
import math
import json
from pathlib import Path
from collections import Counter

def get_dataFrame(argdict):
    """Get the dataframe for the particular split. If it does not exist: create it"""
    create_train=False
    task=argdict['dataset']



    try:
        dfTrain=pd.read_csv(f"{argdict['path']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/train_{argdict['random_seed']}.tsv", sep='\t', index_col=0)
    except:
        create_train=True
    os.makedirs(f"{argdict['path']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/",exist_ok=True)

    if create_train:
        print(f"{argdict['path']}/data/{task}/train.tsv")
        dfTrain=pd.read_csv(f"{argdict['path']}/data/{task}/train.tsv", sep='\t')


        # print(len(dfTrain))
        # fds
    dfVal=pd.read_csv(f'{argdict["path"]}/data/{task}/dev.tsv', sep='\t')
    dfTest=pd.read_csv(f'{argdict["path"]}/data/{task}/test.tsv', sep='\t')
    #
    # if create_train:
    #     #Sampling balanced data
    #     # print(len(dfTrain[dfTrain['label']==0]))
    #     # prop=len(dfTrain[dfTrain['label']==0])/len(dfTrain)
    #     dfTrain['true_label']=dfTrain['label']
    #     nb_points=math.ceil(argdict['dataset_size']/len(argdict['categories']))
    #     # print(prop)
    #     # print(int(argdict['dataset_size']/len(argdict['categories'])))
    #     # print(argdict['labelled_dataset_size'])
    #     NewdfTrain=dfTrain[dfTrain['label']==0].sample(n=nb_points)
    #     for i in range(1, len(argdict['categories'])):
    #         # print(int(argdict['dataset_size']/len(argdict['categories'])))
    #         # print(i)
    #         NewdfTrain=pd.concat([NewdfTrain ,dfTrain[dfTrain['label']==i].sample(n=nb_points)])
    #     dfTrain.drop(NewdfTrain.index)
    #     dfTrain.to_csv(f"{argdict['path']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/train_{argdict['random_seed']}.tsv", sep='\t')
    return dfTrain, dfVal, dfTest


def dataLoaderToArray(dataloader):
    """Takes a dataloader and return an array with the data"""
    x=[]
    y=[]
    # print(dataloader.data)
    # print(len(dataloader.data))
    # print(dataloader.i2w)
    for i in range(len(dataloader.data)):
        data=dataloader.data[i]
        sent = data['sentence']
        x.append(sent)
        y.append(data['label'])
    return x, y
