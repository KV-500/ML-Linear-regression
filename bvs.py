import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

dataset = pd.read_csv('data.csv')
#print(type(dataset))
#dataset.info()
#sns.heatmap(dataset.isnull())
#plt.show()
def impute_sr(cols):
        SR=cols[0]
        BF=cols[1]

        if pd.isnull(SR):

                if BF==0:
                   return 0
                elif BF>=25:
                    return 50
                else:
                    return 30
        else:
            return SR
dataset['SR']=dataset[['SR','BF']].apply(impute_sr,axis=1)
#sns.heatmap(dataset.isnull())
#plt.show()

#print(dataset['SR'])
def impute_mins(cols):
         Mins=cols[0]
         BF=cols[1]

         if pd.isnull(Mins):

             if BF==0:
                 return 0
             elif BF==50:
                 return 50
             else:
                 return 100
         else:
             return Mins
dataset['Mins']=dataset[['Mins','BF']].apply(impute_mins,axis=1)

def impute_runs(cols):
        Runs=cols[0]
        fours=cols[1]

        if pd.isnull(Runs):

             if fours==0:
                 return 0
             elif fours==4:
                 return 4
             else:
                 return 6
        else:
           return Runs
dataset['Runs']=dataset[['Runs','fours']].apply(impute_runs,axis=1)
#sns.heatmap(dataset.isnull())
#plt.show()


dataset.drop('Ground',axis=1,inplace=True)

dataset.dropna(inplace=True)

#sns.heatmap(dataset.isnull())
#plt.show()


dismissal=pd.get_dummies(dataset['Dismissal'],drop_first=True)
#print(dismissal)
result=pd.get_dummies(dataset.Result)
#print(result)

dataset.drop(['Dismissal','Result'],axis=1,inplace=True)

dataset=pd.concat([dataset,dismissal,result],axis=1)
#print(dataset)

X=dataset.iloc[:,1:].values
Y=dataset.iloc[:,0].values
#print(X)
#print(Y)





#BF vs six
fig=plt.figure(figsize=(10,5))
plt.plot('six',color='red')
plt.plot('BF',color='green')
plt.legend(['six','BF'],loc='best',fontsize=20)
plt.title('number of six')
plt.plot(dataset['six'],color='red')
plt.plot(dataset['BF'],color='green')
plt.xlabel('BF', fontsize=30)
plt.ylabel('six',fontsize=30)
plt.show()

















