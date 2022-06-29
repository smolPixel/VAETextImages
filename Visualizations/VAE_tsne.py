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

# Sentences=list(df['sentence'])
# Points=list(df['points'])
# Points=[ast.literal_eval(point) for point in Points]
# # print(Sentences)
# # print(Points)
#
# indOG=2
#
# Comp=Points[indOG]
#
# distances=[distance.cosine(Comp, pp) for pp in Points]
# bests=np.argpartition(distances, 5)[:5]
#
#
# print(bests)
# print(f"Sentence original: {Sentences[indOG]}")
# for i, ind in enumerate(bests):
# 	print(f"Close sentence: {Sentences[ind]}")



plt.scatter(x=xPos, y=yPos, c='red')#, marker='3')
plt.scatter(x=xNeg, y=yNeg, c='blue', marker='3')

plt.savefig(f"../Graphes/test_SST2.png")
#
# dfEx=df[df['x']>75]
# # dfEx=dfEx[dfEx['x']>76]
# dfEx=dfEx[dfEx['y']>40]
# dfEx.to_csv('dfSST2.tsv', sep='\t')
# print(dfEx)