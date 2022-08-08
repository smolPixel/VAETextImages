import pandas as pd

dfTrain=pd.read_csv('train.tsv', sep='\t')
print(dfTrain)

nb_points=50
NewdfTrain=dfTrain[dfTrain['label']==0].sample(n=nb_points)
NewdfTrain=pd.concat([NewdfTrain ,dfTrain[dfTrain['label']==1].sample(n=nb_points)])

NewdfTrain.to_csv('train.tsv', sep='\t')