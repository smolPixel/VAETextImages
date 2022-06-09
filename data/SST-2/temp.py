import pandas as pd

df=pd.read_csv('test.tsv', sep='\t')
for i in df.index.tolist():
	df.at[i, 'true_label']=int(df.at[i, 'label'])
	df.at[i, 'label']=int(df.at[i, 'label'])

print(df)

# file=open('original/sst_test.txt').read().split('\n')[:-1]

# print(all_sentences)


# def process_sentence(sentence):
# 	sentence=sentence.lower()
# 	sentence=sentence.replace(" wanna ", " wan na ")
# 	sentence=sentence.replace(" humour ", " humor ")
# 	sentence=sentence.replace("favourite", "favorite")
# 	sentence=sentence.replace("no. .", "no . .")
# 	sentence=sentence.replace("learnt", "learned")
# 	sentence=sentence.replace("favour", "favor")
# 	return sentence
#
# for line in file:
# 	label, sentence=line.split('\t')
# 	if label=='__label__3':
# 		continue
# 	if label in ['__label__1', '__label__2']:
# 		true_label=0
# 	elif label in ['__label__4', '__label__5']:
# 		true_label=1
# 	else:
# 		raise ValueError
# 	sentence=process_sentence(sentence)
# 	index=all_sentences.index(sentence)
# 	df.at[index, 'label']=true_label
#
df.to_csv('test.tsv', sep='\t', index=False)
