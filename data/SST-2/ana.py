import pandas as pd
import numpy as np

df=pd.read_csv("train.tsv", sep="\t")

sents=list(df['sentence'])
lengths=[len(sent.split(" ")) for sent in sents]
print(np.mean(lengths))

print(len(df))
labels=list(df['label'])
print(sum(labels))

lessFifteen=[ll for ll in lengths if ll<=15]

print(len(lessFifteen))


dfGenerated=pd.read_csv('../../GeneratedData/SST-2/MQS/VAE/2_0.tsv', sep='\t', index_col=0)

print(dfGenerated)
sentenceGen=list(dfGenerated[2:]['sentence'])


tot=len(sentenceGen)
numIn=0
sentIdent=[]
for sen in sentenceGen:
	if sen in sents:
		print(sen)
		sentIdent.append(sen)
		numIn+=1
print(numIn)
print(float(numIn)/tot)
print(np.mean([len(ss.split(" ")) for ss in sentIdent]))