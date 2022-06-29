import pandas as pd
import matplotlib.pyplot as plt
import torch
import ast
from scipy.spatial import distance
import numpy as np
from ast import literal_eval
from sklearn.manifold import TSNE


df=pd.read_csv('../graph_SST2.tsv', sep='\t', index_col=0)
labels=list(df['labs'])

dfPos=df[df['labs']==1]
dfNeg=df[df['labs']==0]


xPos=list(dfPos['x'])
yPos=list(dfPos['y'])
xNeg=list(dfNeg['x'])
yNeg=list(dfNeg['y'])



plt.scatter(x=xPos, y=yPos, c='red')#, marker='3')
plt.scatter(x=xNeg, y=yNeg, c='blue', marker='3')

plt.savefig(f"../Graphes/test_SST2.png")
#
# dfEx=df[df['x']>75]
# # dfEx=dfEx[dfEx['x']>76]
# dfEx=dfEx[dfEx['y']>40]
# dfEx.to_csv('dfSST2.tsv', sep='\t')
# print(dfEx)