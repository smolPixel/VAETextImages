import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk

file=open('wiki.test.tokens', 'r').read().split('\n')

dict={'sentence':[]}

for line in file:
	line=line.strip()
	if line=='' or line[0]=='=':
		continue
	line=sent_tokenize(line)
	for sent in line:
		dict['sentence'].append(sent)


df=pd.DataFrame.from_dict(dict)
df.to_csv('test.tsv', sep='\t')