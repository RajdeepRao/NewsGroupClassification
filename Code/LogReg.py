# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 18:07:11 2016

@author: Rajdeep
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics


frequencyTable=pd.DataFrame()
freqTableTest=pd.DataFrame()

trainDataSubset=pd.read_csv('..\\train6.data', sep=' ', header=None, names=['DocNum','WordNum','Freq'])
trainLablesSubset=pd.read_csv('..\\train6.label', header=None, names=['Lables'])
testDataSubset=pd.read_csv('..\\test6.data', sep=' ', header=None, names=['DocNum','WordNum','Freq'])
testLablesSubset=pd.read_csv('..\\test6.label', header=None, names=['Lables'])

uniqueWordId = np.unique(trainDataSubset['WordNum'])
uniqueDocumentId = np.unique(trainDataSubset['DocNum'])

print('creating freq table')
frequencyTable = pd.DataFrame(index=uniqueDocumentId,columns=uniqueWordId )

temp=pd.DataFrame()
temp=trainDataSubset

for i in temp.itertuples() :
    frequencyTable.loc[i.DocNum,i.WordNum]=i.Freq

frequencyTable.fillna(0,inplace = True)
print("Created Frequency Table for training Data...")
frequencyTable = frequencyTable[frequencyTable.index!= 3388]
frequencyTable.drop(frequencyTable.columns[[range(23765,26760)]], axis=1, inplace=True)

print("Re-shaped the matrix")
logisticRegressionModel=linear_model.LogisticRegression(C=1e5)
logisticRegressionModel.fit(frequencyTable,trainLablesSubset)
print("Trained Logistic Regression Model..")



uniqueWordId2 = np.unique(testDataSubset['WordNum'])
uniqueDocumentId2 = np.unique(testDataSubset['DocNum'])

temp2=pd.DataFrame()
temp2=testDataSubset
freqTableTest = pd.DataFrame(index=uniqueDocumentId2,columns=uniqueWordId2 )
print("Creating Test Freq Table...")
for i in temp2.itertuples() :
    freqTableTest.loc[i.DocNum,i.WordNum]=i.Freq
freqTableTest.fillna(0,inplace = True)
print("Created Test Freq Table...")

logisticRegressionModel.score(frequencyTable,trainLablesSubset)
output=logisticRegressionModel.predict(freqTableTest)
print (metrics.accuracy_score(testLablesSubset, output))
np.savetxt("..\\LogRegOutput.csv",output)
