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

dataset.drop('Pos',axis=1,inplace=True)
#dataset.drop('dismissal',axis=1,inplace=True)
#dataset.drop('Ground',axis=1,inplace=True)
#dataset.drop('result',axis=1,inplace=True)
dataset.drop('Inns',axis=1,inplace=True)
dataset.drop('SR',axis=1,inplace=True)
#dataset.drop('Pos',axis=1,inplace=True)

#dataset=pd.concat([dataset,dismissal,result],axis=1)
print(dataset)

X=dataset.iloc[:,1:].values
Y=dataset.iloc[:,0].values
print(X)
print(Y)




from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)                                              
from sklearn.linear_model import LinearRegression
logmodel=LinearRegression()
logmodel.fit(X_train,Y_train)
pred=logmodel.predict(X_test)
print(X_test)
print(pred)
print(pd.DataFrame(Y_test,pred))
pre1=logmodel.predict([[158,118,6,0]])
print(pre1)

from sklearn.metrics import mean_squared_error
#print(mean_squared_error(Y_test,pred))
#print(logmodel.score(X_test,Y_test))


#from statistics import mean
#BF=dataset.BF
#SR=dataset.SR
#RUNS=pred
#xs=np.array(BF,dtype=np.float64)
#ys=np.array(RUNS,dtype=np.float64)
#def best_fit_slope_and_intercept(xs,ys):
 #       m=(((mean(xs)*mean(ys)) - mean(xs*ys)) /
  #         ((mean(xs)*mean(xs)) - mean(xs*xs)) )
   #     b=mean(ys)-m*mean(xs)

    #    return m, b
#m, b=best_fit_slope_and_intercept(xs,ys)

#regression_line=[(m*x)+b for x in xs]

#plt.title("PLAYER STATE")
#plt.xlabel("BALLS FACED")
#plt.ylabel("RUNS")
#plt.scatter(xs,ys)
#sns.set_style('white')
#sns.set_style('ticks')
#sns.regplot(xs,ys,data=dataset)

#plt.plot(xs,regression_lines)
#plt.show()







