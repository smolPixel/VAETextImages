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
<<<<<<< HEAD
        dfTrain=pd.read_csv(f"{argdict['pathFolder']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['labelled_dataset_size']}/train_{argdict['random_seed']}.tsv", sep='\t', index_col=0)
    except:
        create_train=True
    os.makedirs(f"{argdict['pathFolder']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['labelled_dataset_size']}",exist_ok=True)

    if create_train:
        dfTrain=pd.read_csv(f"{argdict['pathData']}/Dataset/{task}/train.tsv", sep='\t')
=======
        dfTrain=pd.read_csv(f"{argdict['path']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/train_{argdict['random_seed']}.tsv", sep='\t', index_col=0)
    except:
        create_train=True
    os.makedirs(f"{argdict['path']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/",exist_ok=True)

    if create_train:
        dfTrain=pd.read_csv(f"{argdict['path']}/data/{task}/train.tsv", sep='\t')
>>>>>>> f6f8d189e49765d8885a8565af4bb994174e3c3c


        # print(len(dfTrain))
        # fds
<<<<<<< HEAD
    dfVal=pd.read_csv(f'{argdict["pathData"]}/Dataset/{task}/dev.tsv', sep='\t')
    dfTest=pd.read_csv(f'{argdict["pathData"]}/Dataset/{task}/test.tsv', sep='\t')

    if create_train:
        if argdict['labelled_dataset_size']!=0:
            #Sampling balanced data
            # print(len(dfTrain[dfTrain['label']==0]))
            # prop=len(dfTrain[dfTrain['label']==0])/len(dfTrain)
            dfTrain['true_label']=dfTrain['label']
            nb_points=math.ceil(argdict['labelled_dataset_size']/len(argdict['categories']))
            # print(prop)
            # print(int(argdict['dataset_size']/len(argdict['categories'])))
            # print(argdict['labelled_dataset_size'])
            NewdfTrain=dfTrain[dfTrain['label']==0].sample(n=nb_points)
            for i in range(1, len(argdict['categories'])):
                # print(int(argdict['dataset_size']/len(argdict['categories'])))
                # print(i)
                NewdfTrain=pd.concat([NewdfTrain ,dfTrain[dfTrain['label']==i].sample(n=nb_points)])
            dfTrain.drop(NewdfTrain.index)
            dfTrain = dfTrain.assign(label=len(argdict['categories']))
            dfTrain=pd.concat([NewdfTrain, dfTrain])
            dfTrain.to_csv(f"{argdict['pathFolder']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['labelled_dataset_size']}/train_{argdict['random_seed']}.tsv", sep='\t')
        else:
            dfTrain['true_label'] = dfTrain['label']
            dfTrain.to_csv(f"{argdict['pathFolder']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['labelled_dataset_size']}/train_{argdict['random_seed']}.tsv",sep='\t')
=======
    dfVal=pd.read_csv(f'{argdict["path"]}/data/{task}/dev.tsv', sep='\t')
    dfVal=pd.read_csv(f'{argdict["path"]}/data/{task}/dev.tsv', sep='\t')
    dfTest=pd.read_csv(f'{argdict["path"]}/data/{task}/test.tsv', sep='\t')

    if create_train:
        #Sampling balanced data
        # print(len(dfTrain[dfTrain['label']==0]))
        # prop=len(dfTrain[dfTrain['label']==0])/len(dfTrain)
        dfTrain['true_label']=dfTrain['label']
        nb_points=math.ceil(argdict['dataset_size']/len(argdict['categories']))
        # print(prop)
        # print(int(argdict['dataset_size']/len(argdict['categories'])))
        # print(argdict['labelled_dataset_size'])
        NewdfTrain=dfTrain[dfTrain['label']==0].sample(n=nb_points)
        for i in range(1, len(argdict['categories'])):
            # print(int(argdict['dataset_size']/len(argdict['categories'])))
            # print(i)
            NewdfTrain=pd.concat([NewdfTrain ,dfTrain[dfTrain['label']==i].sample(n=nb_points)])
        dfTrain.drop(NewdfTrain.index)
        dfTrain = dfTrain.assign(label=len(argdict['categories']))
        dfTrain=pd.concat([NewdfTrain, dfTrain])
        dfTrain.to_csv(f"{argdict['path']}/SelectedData/{argdict['dataset']}/{argdict['dataset_size']}/train_{argdict['random_seed']}.tsv", sep='\t')
>>>>>>> f6f8d189e49765d8885a8565af4bb994174e3c3c
    return dfTrain, dfVal, dfTest

def process_data(argdict):
    # path=argdict['pathFolder']+"/SelectedData/"+argdict['dataset']+argdict['dataset_size']+'/'
    for files in os.listdir(f"{argdict['pathFolder']}/Generators/{argdict['algo']}/data/"):
        os.remove(f"{argdict['pathFolder']}/Generators/{argdict['algo']}/data/{files}")
    dfTrain, dfDev, dfTest=get_dataFrame(argdict)
    train=""
    trainLabel=""
    dev=""
    devLabel=""
    print(dfTrain)
    for sent in dfTrain['sentence']:
        train+=sent+"\n"
    for lab in dfTrain["label"]:
        trainLabel+=str(lab)+"\n"
    for sent in dfDev['sentence']:
        dev+=sent+"\n"
    for lab in dfDev["label"]:
        devLabel+=str(lab)+"\n"
    file = open(f"{argdict['pathFolder']}/Generators/{argdict['algo']}/data/ptb.train.txt", "w")
    file.write(train)
    file.close()
    file = open(f"{argdict['pathFolder']}/Generators/{argdict['algo']}/data/ptb.train.label.txt", "w")
    file.write(trainLabel)
    file.close()
    file = open(f"{argdict['pathFolder']}/Generators/{argdict['algo']}/data/ptb.dev.txt", "w")
    file.write(dev)
    file.close()
    file = open(f"{argdict['pathFolder']}/Generators/{argdict['algo']}/data/ptb.dev.label.txt", "w")
    file.write(devLabel)
    file.close()

<<<<<<< HEAD
    # #Baseline
    # olddsSize=argdict['labelled_dataset_size']
    # argdict['labelled_dataset_size']=argdict['labelled_dataset_size']+argdict["batch_size_iter"]*argdict["nb_iter"]
    # dfTrain, dfDev=get_dataFrame(argdict)
    # train=""
    # trainLabel=""
    # dev=""
    # devLabel=""
    # for sent in dfTrain['sentence']:
    #     train+=sent+"\n"
    # for lab in dfTrain["label"]:
    #     trainLabel+=str(lab)+"\n"
    # for sent in dfDev['sentence']:
    #     dev+=sent+"\n"
    # for lab in dfDev["label"]:
    #     devLabel+=str(lab)+"\n"
    # os.makedirs(f"{argdict['pathFolder']}/Generators/{argdict['algo']}/Databaseline", exist_ok=True)
    # file = open(f"{argdict['pathFolder']}/Generators/{argdict['algo']}/Databaseline/ptb.train.txt", "w")
    # file.write(train)
    # file.close()
    # file = open(f"{argdict['pathFolder']}/Generators/{argdict['algo']}/Databaseline/ptb.train.label.txt", "w")
    # file.write(trainLabel)
    # file.close()
    # file = open(f"{argdict['pathFolder']}/Generators/{argdict['algo']}/Databaseline/ptb.dev.txt", "w")
    # file.write(dev)
    # file.close()
    # file = open(f"{argdict['pathFolder']}/Generators/{argdict['algo']}/Databaseline/ptb.dev.label.txt", "w")
    # file.write(devLabel)
    # file.close()
    # argdict['labelled_dataset_size']=olddsSize
=======
>>>>>>> f6f8d189e49765d8885a8565af4bb994174e3c3c

def add_data(argdict, sentences, labels, datasetsLabelled):
    """Add labelled data to the algo"""
    # print(sentences)
    # print(labels)
    datasetsLabelled['train'].add_data(sentences, labels)
    return datasetsLabelled

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

<<<<<<< HEAD
def compareFiles(argdict, file, folder):
    """Compare the argdict and the string of parameter EXCEPT for the computer parameter
    folder can be gen or exp, for generated or experiments"""

    # if folder=="gen":
    #     # notInt=["computer", "retrain", "rerun", "run_ds_split", "test_latent_size_vs_split", "pathDataOG", "pathDataAdd", "categories", "cat", "nb_epoch_lstm",
    #     #         "dropout_classifier", "hidden_size_classifier", "numFolderGenerated", "classifier", "split"]
    #     #Instead what is important
    #     In=ImportantAlgo
    # else:
    #     # notInt=["computer", "retrain", "rerun", "run_ds_split", "test_latent_size_vs_split", "pathDataOG", "pathDataAdd", "categories", "cat",
    #     #         "test_hidden_size_algo_vs_split", "numFolderGenerated"]
    #     #Everything that is important for generating the dataset makes a difference in classification
    #     In=ImportantClassifier
    # print("---------")
    for key, value in argdict.items():
        # if key in In:

        try:
            value2=file[key]
            if value2!=value:
                return False
        except:
            return False
    return True

def checkFoldersGenerated(argdict):
    """Check in Results if the experiment has not been done already. If not, create new experiments number"""
    found=False
    i=0
    while not found:
        path=f"{argdict['pathFolder']}/GeneratedData/{argdict['dataset']}/{argdict['processus']}/{argdict['algo']}/{i}"
        try:
            #Open the folder. If it works then check if its the same experiment, else go to next one
            param=json.load(open(f"{path}/param.json", 'r'))
            sentences=pd.read_csv(f"{path}/sentences.tsv", sep='\t')
            if compareFiles(argdict, param, "exp"):
                print(f"Dataset was already generated in folder {i}")
                # print(f"Results are {results}")
                found=True
                return sentences, i
            i+=1
        except:
            #We could not open this folder, aka empty experiment. Create folder
            print("Creating new folder")
            Path(path).mkdir(parents=True, exist_ok=True)
            return False, i

def checkFolders(argdict):
    """Check in Results if the experiment has not been done already. If not, create new experiments number"""
    found=False
    i=0
    while not found:
        path=f"{argdict['pathFolder']}/Results/{argdict['dataset']}/{argdict['processus']}/{argdict['algo']}/{i}"
        try:
            #Open the folder. If it works then check if its the same experiment, else go to next one
            param=json.load(open(f"{path}/param.json", 'r'))
            results=json.load(open(f"{path}/out.json", 'r'))
            if compareFiles(argdict, param, "exp"):
                print(f"Experiment was already run in folder {i}")
                # print(f"Results are {results}")
                found=True
                return results, i
            i+=1
        except:
            #We could not open this folder, aka empty experiment. Create folder
            print("Creating new folder")
            Path(path).mkdir(parents=True, exist_ok=True)
            return False, i

def log_results(argdict, results, i):
    path = f"{argdict['pathFolder']}/Results/{argdict['dataset']}/{argdict['processus']}/{argdict['algo']}/{i}"
    #Log results
    with open(f"{path}/param.json", "w") as f:
        json.dump(argdict, f)
    with open(f"{path}/out.json", "w") as f:
        json.dump(results, f)
    # path = f"{argdict['pathDataAdd']}/GeneratedData/{argdict['algo']}/{argdict['dataset']}/{argdict['dataset_size']}/{argdict['numFolderGenerated']}"
    # with open(f"{path}/param.json", "w") as f:
    #     json.dump(argdict, f)
=======


>>>>>>> f6f8d189e49765d8885a8565af4bb994174e3c3c
