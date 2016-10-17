# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 16:28:08 2016

@author: Rajdeep
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.cross_validation import train_test_split

trainDataIJV=pd.read_csv('..n\\train.data',sep=' ', header=None, names=['DocNum','WordNum','Freq'])
testDataIJV=pd.read_csv('..\\test.data', sep=' ', header=None, names=['DocNum','WordNum','Freq'])
trainLables=pd.read_csv('..\\train.label', header=None, names=['Lables'])
testLables=pd.read_csv('..\\test.label')

maxWord1=trainDataIJV.max()
maxWord2=testDataIJV.max()

maxWord=max(maxWord2['WordNum'],maxWord2['WordNum'] )
#trainData=sp.sparse.csr_matrix((trainDataIJV.values[3],(trainDataIJV.values[2],trainDataIJV.values[1])), shape=(maxWord, max(testDataIJV[1])) )
#print(trainData)

trainDataIJV['Freq']= np.where(trainDataIJV['Freq']>0, 1,0)
testDataIJV['Freq']= np.where(testDataIJV['Freq']>0, 1,0)
freqTable=pd.DataFrame()
freqTableTest=pd.DataFrame()

def naiveBayesianTrain(trainDataIJV, trainLables, params):
    temp=pd.DataFrame()
    temp=trainDataIJV.query('DocNum<500')

    for i in temp.itertuples() :
        freqTable.loc[i.DocNum,i.WordNum]=1

    freqTable.fillna(0,inplace = True)
    print("Created Frequency Table for training Data...")

    classifier=MultinomialNB(alpha=1.0)
    classifier.fit(freqTable, trainLables.query('index<499'))
    print("Trained Naive Bayesian Model...")
    return classifier

def naiveBayesianTest(testDataIJV,model):

    temp2=pd.DataFrame()
    temp2=testDataIJV.query('DocNum<500')
    for i in temp2.itertuples() :
        freqTableTest.loc[i.DocNum,i.WordNum]=1

    freqTableTest.fillna(0,inplace = True)
    print("Created Test Freq Table...")
    freqTableTest.drop(freqTableTest.columns[[range(10275,12866)]], axis=1, inplace=True)
    output=model.predict(freqTableTest[0:499])
    np.savetxt('..\\NBTestDataOutput.csv',output)
    return 0

def decisionTreeTrain(trainDataIJV, trainLabels):

    classifier2 = tree.DecisionTreeClassifier(max_depth=100)
    classifier2 = classifier2.fit(freqTable, trainLables.query('index<499'))
    print("Trained Decision Tree Model...")
    return classifier2

def decisionTreeTest(testDataIJV,model2):
    output2=model2.predict(freqTableTest[0:499])
    np.savetxt('..\\DTTestDataOutput.csv',output2)
    return 0

def crossValidation(trainAlgo,predictAlgo,trainDataIJV,trainLables,folds):

    temp3=pd.DataFrame()
    temp3=trainLables.query('index<499')
    predictedVal=pd.DataFrame()
    predictedVal2=pd.DataFrame()
    sum1=0
    sum2=0
    classifierDT=tree.DecisionTreeClassifier(max_depth=10)
    classifierNB=MultinomialNB(alpha=1.0)

    if trainAlgo==1:
        predictedVal=pd.DataFrame()
        sum1=0

        for i in range(1,folds):
           k=-1
           xtrain, xtest, ytrain, ytest = train_test_split(freqTable, temp3, test_size=0.10)
           classifierNB.fit(xtrain, ytrain)
           predictedVal=classifierNB.predict(xtest)
           for j in ytest.itertuples():
               k=k+1
               if predictedVal[k]==j.Lables:
                   sum1=sum1+1

           score=sum1*100/500
           print('The Score is: ' + str(score) + ' % match')

    else:
        predictedVal2=pd.DataFrame()
        sum2=0

        for i in range(1,folds):
           k=-1
           xtrain, xtest, ytrain, ytest = train_test_split(freqTable, temp3, test_size=0.10)
           classifierDT.fit(xtrain, ytrain)
           predictedVal2=classifierDT.predict(xtest)
           for j in ytest.itertuples():
               k=k+1
               if predictedVal2[k]==j.Lables:
                   sum2=sum2+1

           score=sum2*100/500
           print('The Score is: ' + str(score) + ' % match')

    return 0

#Main Sequence of implementation Logic


model=naiveBayesianTrain(trainDataIJV,trainLables,5)
call1=naiveBayesianTest(testDataIJV,model)
print(call1)

model2=decisionTreeTrain(trainDataIJV, trainLables)
call2=decisionTreeTest(testDataIJV,model2)
print(call2)

call3=crossValidation(1,1,trainDataIJV,trainLables,10)
print(call3)




#This code may not work very well, but the logic is about right, because I implemented this for 500 documents and calculated the gain succeddfully
def calculateInfoGain(trainDataIJV, trainLabels):

    uniqueWordNum = np.unique(trainDataIJV['WordNum'])
    uniqueDocumentNum = np.unique(trainDataIJV['DocNum'])
    totalDocNumber = len(uniqueDocumentNum)
    dataGroup = pd.DataFrame()
    frequencyTable = pd.DataFrame(index=uniqueDocumentNum,columns=uniqueWordNum )

    for j in trainDataIJV.itertuples():
        frequencyTable.loc[j[1],j[2]] = 1

    frequencyTable.fillna(0,inplace=True)
    frequencyTable.reset_index(inplace= True)
    frequencyTable.rename(columns={'index':'DocNum'},inplace= True)
    frequencyTable=frequencyTable.merge(trainLabels,on='Index',how='left')
    dataGroup=frequencyTable.copy()
    dataGroup['constant']=1
    dataGroup=dataGroup.groupby(['classId'],as_index=False).sum()
    totalEntropy=0
    informationGain=pd.Series(name='infoGain')

    for i in np.unique(dataGroup['classId']):
        classCount=dataGroup.loc[i-1,'constant']
        div=classCount/totalDocNumber
        totalEntropy=totalEntropy+(-(div)*np.log(div))

    for i in np.arange(1,trainDataIJV['wordNum'].max()+1):
        count1=frequencyTable[1].sum()
        count0=len(frequencyTable[1])-count1
        df=pd.DataFrame(index=[0,1],columns= np.arange(1,21))
        df.fillna(0,inplace=True)
        df.loc[1]=dataGroup[i].reshape(1,20)
        df.loc[0]=dataGroup['constant'].reshape(1,20)-dataGroup[i].reshape(1,20)
        sum1=0
        sum0=0
        attributEntropy=0
        for j in np.unique(dataGroup['classId']):
            sum1Div=df.loc[1,j]/count1
            sum0Div=df.loc[0,j]/count0
            if sum1Div!=0:
                sum1=sum1+(-(sum1Div)*np.log((sum1Div)))
            if sum0Div!=0:
                sum0=sum0+(-(sum0Div)*np.log((sum0Div)))

        attributEntropy = -((count1/totalDocNumber)*sum1) -((count0/totalDocNumber)*sum0)
        informationGain.loc[i]= totalEntropy + attributEntropy

    informationGain.to_csv('..\\informationGain.csv')



calculateInfoGain(trainDataIJV,trainLables)
