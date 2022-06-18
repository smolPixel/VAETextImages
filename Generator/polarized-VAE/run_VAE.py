"""This file is in charge of running the VAE to generate positif and negatif examples, as well as running experiments on it
To do so it
1- Create ptb files for pos, train VAE, generate, repeat with neg
2-
"""

import subprocess
import argparse
import os
import shutil
import json

import run_glue_xgboost
import run_glue_lstm
from process_data import process_data_for_SVAE, add_data, createFoldersSVAE, checkTrained

def main(argdict):
    #Check whether we should retrain
    argdict=checkTrained(argdict)
    data=argdict['dataset']
    split=argdict['split'] #Percent of data points added
    dataset_size=argdict['dataset_size']
    # Delete old generated data
    if argdict['retrain']:
        for files in os.listdir(f'GeneratedData/{data}/{dataset_size}'):
            os.remove(f'GeneratedData/{data}/{dataset_size}/{files}')

    for pola in ['pos', 'neg']:
        if argdict['retrain']:
            for files in os.listdir('data'):
                os.remove(f'data/{files}')
            for direc in os.listdir('bin'):
                shutil.rmtree(f"bin/{direc}", ignore_errors=True)
        argdict['polarity']=pola
        num_points=process_data_for_SVAE(argdict)
        # print(num_points)
        num_points = int(num_points)
        # Generate the maximum number of points, that is 5 times the dataset per class
        num_generated = round(num_points * 5)
        argdict['num_to_add'] = round(num_points * split / 2)
        if argdict['retrain']:
            # #Training VAE
            bashCommand = f'python3 train.py -ep {argdict["nb_epoch_VAE"]}'
            process = subprocess.Popen(bashCommand.split())
            output, error = process.communicate()
            #Inference
            for dir in os.listdir('bin'):
                #TODO CHANGE THIS IF MORE THAN 9
                dir=f'bin/{dir}/E9.pytorch'
            # Prevent memory overflow
            max_gen=5000
            while num_generated > max_gen:
                num_gen_cur=max_gen
                num_generated=num_generated-max_gen
                bashCommand = f'python3 inference.py -c {dir} -n {num_gen_cur}' #GeneratedData/{data}/{pola}.txt'

                with open(f"GeneratedData/{data}/{dataset_size}/{pola}.txt", "a") as outfile:
                    process = subprocess.Popen(bashCommand.split(), stdout=outfile)
                    output, error = process.communicate()
            bashCommand = f'python3 inference.py -c {dir} -n {num_generated}'  # GeneratedData/{data}/{pola}.txt'

            with open(f"GeneratedData/{data}/{dataset_size}/{pola}.txt", "a") as outfile:
                process = subprocess.Popen(bashCommand.split(), stdout=outfile)
                output, error = process.communicate()

    #Add data to dataframe
    add_data(argdict, "VAE")
    if argdict['classifier']=='jiantLstm':
        bashCommand = f'python3 ../run_glue_jiantLSTM.py --dataset {argdict["dataset"]}' #GeneratedData/{data}/{pola}.txt'
        process = subprocess.Popen(bashCommand.split())
        output, error = process.communicate()
    elif argdict['classifier']=='lstm':
        acc_train, acc_val=run_glue_lstm.run_lstm(argdict)
    elif argdict['classifier']=='xgboost':
        acc_train, acc_val = run_glue_xgboost.run_xgboost(argdict)
    return acc_train, acc_val






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE for data augmentation')
    #General arguments on training
    parser.add_argument('--dataset', type=str, default='SST-2', help="dataset you want to run the process on. Includes SST2, CoLA, MRPC")
    parser.add_argument('--classifier', type=str, default='lstm', help="classifier you want to use. Includes bert, lstm, jiantLstm or xgboost")
    parser.add_argument('--computer', type=str, default='home', help="Whether you run at home or at iro. Automatically changes the base path")
    parser.add_argument('--split', type=float, default=1.0, help='percent of the dataset added')
    parser.add_argument('--retrain', action='store_true', help='whether to retrain the VAE or not')
    parser.add_argument('--dataset_size', type=int, default=0, help='number of example in the original dataset. If 0, use the entire dataset')

    #VAE specific arguments
    parser.add_argument('--nb_epoch_VAE', type=int, default=10, help="Number of epoch for which to run the VAE")

    #Experiments
    parser.add_argument('--test_dss_vs_split', action='store_true', help='test the influence of the dataset size and ')

    args = parser.parse_args()

    argsdict = args.__dict__
    argsdict['pathIn'] = "/media/frederic/DAGlue/jiant-v1-legacy/data"
    argsdict['pathOut'] = "/media/frederic/DAGlue/data"
    # Create directories for the runs
    createFoldersSVAE(argsdict)
    json.dump(argsdict, open(f"/media/frederic/DAGlue/SentenceVAE/GeneratedData/{argsdict['dataset']}/{argsdict['dataset_size']}/parameters.json", "w"))


    print("=================================================================================================")
    print(argsdict)
    print("=================================================================================================")

    if argsdict['test_dss_vs_split']:
        dico={'help':'key is dataset size, then key inside is split and value is a tuple of train val accuracy'}
        argsdict['nb_epoch_VAE']=20
        for ds in [10,50,100,500,5000,10000,20000,30000,40000,50000]:
            dicoTemp={}
            for split in [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                print(f"Dataset size {ds}, split {split}")
                argsdict['dataset_size']=ds
                argsdict['split']=split
                argsdict['retrain']=False
                createFoldersSVAE(argsdict)
                acc_train, acc_val=main(argsdict)
                dicoTemp[split]=(acc_train, acc_val)
            dico[ds]=dicoTemp
            with open(f"/media/frederic/DAGlue/Experiments/DSSVSSplit/VAE_{argsdict['classifier']}.json", "w") as f:
                json.dump(dico, f)
    else:
        main(argsdict)