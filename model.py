import numpy as np
import pandas as pd
import seaborn as sns
import joblib

df = pd.read_csv("agridata_csv_202110311352.csv",na_values='=')
data2 = df.copy()
data2 = data2.dropna()
data2 = data2.head(800000)

str = data2["date"][1]
str2 = str.split('-')

Dict = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
month_column = []

for rr in data2["date"]:
    str = rr
    str2 = str.split('-')
    month_column.append(Dict[int(str2[1])])

data2["month_column"]=month_column

season_names=[]
for tt in data2["month_column"]:
    if tt=="January" or tt=="February":
        season_names.append("Winter")
    elif tt=="March" or tt=="April":
        season_names.append("Spring")
    elif tt=="May" or tt=="June":
        season_names.append("Summer")
    elif tt=="July" or tt=="August":
        season_names.append("Monsoon")
    elif tt=="September" or tt=="October":
        season_names.append("Autumn")
    elif tt=="November" or tt=="December":
        season_names.append("Pre winter")

data2["season_names"]=season_names

day_of_week=[]

for rr in data2["date"]:
    str = rr
    df = pd.Timestamp(rr)
    day = df.dayofweek
    day_of_week.append(day)

data2["day"]=day_of_week

data2=data2.drop('date',1)

data2=data2.head(100000)

#IQR
Q1 = np.percentile(data2['modal_price'], 25,interpolation = "midpoint")

Q3 = np.percentile(data2['modal_price'], 75,interpolation = "midpoint")
IQR = Q3 - Q1

#Upper bound
upper = np.where(data2['modal_price'] >= (Q3+1.5*IQR))
#Lower bound
lower = np.where(data2['modal_price'] >= (Q3-1.5*IQR))

df = data2.copy()
data3 = df.copy()

import plotly.express as px

dist = (data2['commodity_name'])
mydist = (data3['commodity_name'])
distset=set(dist)
mydistset=set(mydist)
dd= list(distset)
mydd= list(mydistset)
dictOfWordsA = {dd[i] : i for i in range(0,len(dd))}
mydictOfWordsA = {mydd[i] : i for i in range(0,len(mydd))}
data2['commodity_name']=data2['commodity_name'].map(dictOfWordsA)

dist = (data2['state'])
mydist = (data3['state'])
# print(data2['state'])
distset=set(dist)
mydistset=set(mydist)
dd= list(distset)
mydd= list(mydistset)
dictOfWordsB = {dd[i] : i for i in range(0,len(dd))}
mydictOfWordsB = {mydd[i] : i for i in range(0,len(mydd))}
data2['state']=data2['state'].map(dictOfWordsB)

dist = (data2['district'])
mydist = (data3['district'])
distset=set(dist)
mydistset=set(mydist)
dd= list(distset)
mydd= list(mydistset)
dictOfWordsC = {dd[i] : i for i in range(0,len(dd))}
mydictOfWordsC = {mydd[i] : i for i in range(0,len(mydd))}
data2['district']=data2['district'].map(dictOfWordsC)

dist = (data2['market'])
my_dist = (data3['market'])
distset=set(dist)
mydistset = set(my_dist)
dd= list(distset)
mydd = list(mydistset)
dictOfWordsD = {dd[i] : i for i in range(0,len(dd))}
mydictOfWordsD = {mydd[i] : i for i in range(0,len(mydd))}
data2['market']=data2['market'].map(dictOfWordsD)

dist = (data2['month_column'])
mydist = (data3['month_column'])
distset=set(dist)
mydistset=set(mydist)
dd= list(distset)
mydd= list(mydistset)
dictOfWordsE = {dd[i] : i for i in range(0,len(dd))}
mydictOfWordsE = {mydd[i] : i for i in range(0,len(mydd))}
data2['month_column']=data2['month_column'].map(dictOfWordsE)

dist = (data2['season_names'])
distset=set(dist)
dd= list(distset)
dictOfWordsF = {dd[i] : i for i in range(0,len(dd))}
mydist = (data3['season_names'])
mydistset=set(mydist)
mydd= list(mydistset)
mydictOfWordsF = {mydd[i] : i for i in range(0,len(mydd))}
data2['season_names']=data2['season_names'].map(dictOfWordsF)

features = data2[["commodity_name", "state","district","market","month_column","season_names","day"]]
labels = data2['modal_price']

#Splitting into train and test data
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(features,labels,test_size=0.2,random_state=2)

from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

regr = RandomForestRegressor(max_depth=1000, random_state=0)
regr.fit(Xtrain,Ytrain)

y_pred = regr.predict(Xtest)

from sklearn.metrics import r2_score

r2_score(Ytest,y_pred)

def convert_to_num(a,b,c,d,e,f):
#     print(mydictOfWordsB)
    result = []
    for key in mydictOfWordsA:
        if key == a:
            result.append(mydictOfWordsA.get(key))
    for key in mydictOfWordsB:
        if key == b:
            result.append(mydictOfWordsB.get(key))
    for key in mydictOfWordsC:
        if key == c:
            result.append(mydictOfWordsC.get(key))
    for key in mydictOfWordsD:
        if key == d:
            result.append(mydictOfWordsD.get(key))
    for key in mydictOfWordsE:
        if key == e:
            result.append(mydictOfWordsE.get(key))
    for key in mydictOfWordsF:
        if key == f:
            result.append(mydictOfWordsF.get(key))
    return(result)
# result = convert_to_num(a,b,c,d,e,f)
# result.append(g)
# result = [result,]

def answer(result):
    return regr.predict(result)

filename = "final.joblib"
joblib.dump(regr,filename)

# print(regr.predict(result))


    

# user_input=convert_to_num(a,b,c,d)
# regr.predict(user_input)

