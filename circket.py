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
print(dataset)

X=dataset.iloc[:,1:].values
Y=dataset.iloc[:,0].values
print(X)
print(Y)

'''
#BF vs Runs
fig=plt.figure(figsize=(263,15))
plt.plot('Runs',color='red')
plt.plot('BF',color='green')
plt.legend(['Runs','BF'],loc='best',fontsize=20)
plt.title('BF vs Runs')
plt.plot(dataset['Runs'])
plt.plot(dataset['BF'])
plt.xlabel('BF', fontsize=30)
plt.ylabel('Runs',fontsize=30)
plt.show()

#fours
fig=plt.figure(figsize=(263,15))
plt.plot('Runs',color='red')
plt.plot('BF',color='green')
plt.legend(['Runs','BF'],loc='best',fontsize=20)
plt.title('Fours')
plt.plot(dataset['fours'])
plt.xlabel('BF', fontsize=30)
plt.ylabel('Fours',fontsize=30)
plt.show()

#BF vs fours
fig=plt.figure(figsize=(263,15))
plt.plot('fours',color='red')
plt.plot('BF',color='green')
plt.legend(['fours','BF'],loc='best',fontsize=20)
plt.title('number of fours')
plt.plot(dataset['fours'],color='red')
plt.plot(dataset['BF'],color='green')
plt.xlabel('BF', fontsize=30)
plt.ylabel('fours',fontsize=30)
plt.show()

#six
fig=plt.figure(figsize=(263,15))
plt.plot('Runs',color='red')
plt.plot('BF',color='green')
plt.legend(['Runs','BF'],loc='best',fontsize=20)
plt.title('six')
plt.plot(dataset['six'])
plt.xlabel('BF', fontsize=30)
plt.ylabel('Six',fontsize=30)
plt.show()


#BF vs six
fig=plt.figure(figsize=(263,15))
plt.plot('six',color='red')
plt.plot('BF',color='green')
plt.legend(['six','BF'],loc='best',fontsize=20)
plt.title('number of six')
plt.plot(dataset['six'],color='red')
plt.plot(dataset['BF'],color='green')
plt.xlabel('BF', fontsize=30)
plt.ylabel('six',fontsize=30)
plt.show()







fig=plt.figure(figsize=(263,15))
plt.title('PIE')
FOURS=sum(dataset.fours)
SIX=sum(dataset.six)
RUNS=(sum(dataset.Runs)-(FOURS+SIX))
values=[FOURS,SIX,RUNS]
label=["Fours","Six","Runs"]
plt.pie(values,labels=label)
plt.show()


#Runs
range=[10,20,30,40,50,60,70,80,90,100]
fig=plt.hist(dataset.Runs,range,histtype='bar',color='darkslategrey',rwidth=0.5)
plt.xlabel("Range of Runs",size=30)
plt.ylabel("No. of times",size=30)
plt.title("Runs")
plt.show()

plt.figure(figsize=(20,10))
plt.barh(dataset.fours,dataset.six)
plt.title("fours & six",size=30)
plt.show()
'''


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)                                              
from sklearn.linear_model import LinearRegression
logmodel=LinearRegression()
logmodel.fit(X_train,Y_train)
pred=logmodel.predict(X_test)
print(X_test)
print(pred)
print(pd.DataFrame(Y_test,pred))

from sklearn.metrics import mean_squared_error
print(mean_squared_error(Y_test,pred))
print(logmodel.score(X_test,Y_test))









