"""Labellers remplace the oracle in this version"""
import torch
import pandas as pd
# from Labellers.labeller_template import labeller_template

class classifier():

    def __init__(self, argdict):
        print(argdict)
        if argdict['classifier']=='svm':
            from Classifiers.svm import SVM_Classifier
            self.algo=SVM_Classifier(argdict)
        elif argdict['classifier']=='LogReg':
            from Classifiers.LogReg import LogReg_Classifier
            self.algo=LogReg_Classifier(argdict)
        elif argdict['classifier']=="bert":
            from Classifiers.Bert import Bert_Classifier
            self.algo=Bert_Classifier(argdict)
        elif argdict['classifier'].lower()=="bert_calibrated":
            from Classifiers.Bert_Calibrated import Bert_Calibrated_Classifier
            self.algo=Bert_Calibrated_Classifier(argdict)
        elif argdict['classifier']=="jiant":
            from Classifiers.jiant import Jiant_Classifier
            self.algo=Jiant_Classifier(argdict)
        elif argdict['classifier']=='svm_latent':
            from Classifiers.svm_latent import SVM_Latent_Classifier
            self.algo=SVM_Latent_Classifier(argdict, datasets, datasetsLabelled)
        else:
            raise ValueError(f"No classifier named {argdict['classifier']}")
        self.argdict=argdict
        # self.trainData, self.devData= self.load_data()


    def init_model(self):
        self.algo.init_model()


    def train(self, train, dev):
        return self.algo.train_model(train, dev)

    def train_test(self, datasetTrain, datasetDev, datasetTest, num_epochs=-1):
        """Receive as argument a dataloader from pytorch"""
        return self.algo.train_model(datasetTrain, datasetDev, datasetTest, num_epochs)