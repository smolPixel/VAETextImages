import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score,matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

#Tasks in SST-2, CoLA, MNLI, MRPC, QNLI, QQP, RTE, SNLI, STS-B, WNLI

task="SST-2"

def read_fucked_files(file, skip_header=False):
    #Take in a file name and return an array of array containing the columns in order
    fp=open(file, 'r')
    arr=None
    for j, line in enumerate(fp.readlines()):
        if j==0 and skip_header:
            continue
        line=line.split('\t')
        if arr is None:
            arr=[]
            for _ in line:
                arr.append([])
        for i, ll in enumerate(line):
            arr[i].append(ll)
    return arr

def fuse_sentences(L1, L2):
    arr=[]
    for l1, l2 in zip(L1, L2):
        arr.append(l1+" [SEP] "+l2)
    return arr

if task in ["SST-2",  "QQP", "RTE", "WNLI"]:
    dfTrain=pd.read_csv(f'../jiant-v1-legacy/data/{task}/train.tsv', sep='\t')
    dfVal=pd.read_csv(f'../jiant-v1-legacy/data/{task}/dev.tsv', sep='\t')
elif task in ["CoLA"]:
    dfTrain = pd.read_csv(f'../jiant-v1-legacy/data/{task}/train.tsv', names=["", "label", "s", "sentence"], header=None, sep='\t')
    dfVal = pd.read_csv(f'../jiant-v1-legacy/data/{task}/dev.tsv',names=["", "label", "s", "sentence"],  header=None, sep='\t')
elif task in ["MRPC", "STS-B", "QNLI"]:
    dfTrain=read_fucked_files(f'../jiant-v1-legacy/data/{task}/train.tsv', skip_header=True)
    dfVal=read_fucked_files(f'../jiant-v1-legacy/data/{task}/dev.tsv', skip_header=True)
elif task in ["MNLI"]:
    dfTrain=read_fucked_files(f'../jiant-v1-legacy/data/{task}/train.tsv', skip_header=True)
    dfVal=read_fucked_files(f'../jiant-v1-legacy/data/{task}/dev_matched.tsv', skip_header=True)
    dfValMis=read_fucked_files(f'../jiant-v1-legacy/data/{task}/dev_mismatched.tsv', skip_header=True)


if task in ["SST-2", "CoLA"]:
    XTrain=dfTrain['sentence']
    YTrain=dfTrain['label']
    XPos=XTrain[YTrain==1]
    XNeg=XTrain[YTrain==0]
    XVal=dfVal['sentence']
    YVal=dfVal['label']
elif task in ["MRPC"]:
    XTrain=fuse_sentences(dfTrain[3], dfTrain[4])
    YTrain=[int(i) for i in dfTrain[0]]
    XVal=fuse_sentences(dfVal[3], dfVal[4])
    YVal=[int(i) for i in dfVal[0]]
elif task in ["QNLI"]:
    XTrain=fuse_sentences(dfTrain[1], dfTrain[2])
    YTrain=[i.strip() for i in dfTrain[3]]
    XVal=fuse_sentences(dfVal[1], dfVal[2])
    YVal=[i.strip() for i in dfVal[3]]
elif task in ["STS-B"]:
    XTrain=fuse_sentences(dfTrain[7], dfTrain[8])
    YTrain=[float(i) for i in dfTrain[9]]
    XVal=fuse_sentences(dfVal[7], dfVal[8])
    YVal=[float(i) for i in dfVal[9]]
elif task in ["QQP"]:
    XTrain=fuse_sentences(dfTrain["question1"], dfTrain["question2"])
    YTrain=dfTrain["is_duplicate"]
    XVal = fuse_sentences(dfVal["question1"], dfVal["question2"])
    YVal = dfVal["is_duplicate"]
elif task in ["RTE", "WNLI"]:
    XTrain=fuse_sentences(dfTrain["sentence1"], dfTrain["sentence2"])
    YTrain=list(dfTrain["label"])
    XVal = fuse_sentences(dfVal["sentence1"], dfVal["sentence2"])
    YVal = list(dfVal["label"])
elif task in ["MNLI"]:
    XTrain=fuse_sentences(dfTrain[8], dfTrain[9])
    YTrain=dfTrain[11]
    XVal=fuse_sentences(dfVal[8], dfVal[9])
    YVal=dfVal[15]
    XValMimatched = fuse_sentences(dfValMis[8], dfValMis[9])
    YValMismatched =dfValMis[15]




trainFile=""
valFile=""
for line in XNeg:
    trainFile+=line+"\n"
for line in XVal:
    valFile+=line+"\n"

file=open("data/ptb.train.txt", "w")
file.write(trainFile)
file.close()
file=open("data/ptb.valid.txt", "w")
file.write(valFile)
file.close()