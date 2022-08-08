"""Finding out the NLL for a n-gram model"""

import pandas as pd


class n_grams_model():

    def __init__(self, n) -> None:
        self.n=n
        self.dico={}


    def fit(self, corpus):
        for sent in corpus:
            sent=sent.split(" ")
            print(sent)
            for i in range(len(sent)-self.n+1):
                key=sent[i]
                value=sent[i+1]
                if key in self.dico.keys():
                    if value in self.dico[key].keys():
                        self.dico[key][value]+=1
                    else:
                        self.dico[key][value]=1
                else:
                    self.dico[key]={sent[i+1]:1}
    


dfTrain=pd.read_csv("train.tsv", sep='\t')
sents_train=list(dfTrain['sentence'])
ngram=n_grams_model(2)
ngram.fit(sents_train)
print(ngram.dico)