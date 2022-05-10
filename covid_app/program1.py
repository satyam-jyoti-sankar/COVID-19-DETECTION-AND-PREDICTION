import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from statsmodels.tsa.api import Holt

import base64
from io import BytesIO

from pathlib import Path
import os

# for directory path
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = os.path.join(BASE_DIR,'templates')
STATIC_ROOT = os.path.join(BASE_DIR, 'csv_files')
COVID_FILE_PATH = STATIC_ROOT+'\covid_new_data.csv'


covid = pd.read_csv(COVID_FILE_PATH)
# covid.tail(4)

# no print
# print("Size/Shape of the dataset",covid.shape)
# print("Checking for null values",covid.isnull().sum())
# print("Checking Data-type",covid.dtypes)

# no print
# print(covid["ObservationDate"][-5:])

# def showfiledata():
#     filedata = covid.tail(5)
#     return filedata


#Grouping differnent types of cases as per the date
covid["ObservationDate"] = pd.to_datetime(covid["ObservationDate"])
datewise = covid.groupby(["ObservationDate"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})


# print this
# print("Basic Information")
# print('last 5 days case:-',datewise["ObservationDate"])
# print("Total number of Confirmed cases around the world",datewise["Confirmed"].iloc[-1])
# print("Total number of Recovered cases around the world",datewise["Recovered"].iloc[-1])
# print("Total number of Death cases around the world",datewise["Deaths"].iloc[-1])
# print("Total number of Active cases around the world",(datewise["Confirmed"].iloc[-1]-datewise["Recovered"].iloc[-1]-datewise["Deaths"].iloc[-1]))
# print("Total number of Closed cases around the world",(datewise["Recovered"].iloc[-1]+datewise["Deaths"].iloc[-1]))

# print("Total number of Confirmed cases around the world",datewise["Confirmed"].tail(15))
# print(covid["ObservationDate"].tail(15))



def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png= buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

#this graph belongs to last 60 days active case
# plt.figure(figsize=(15,5))
# sns.barplot(x=datewise.index.date[-60:],y=datewise["Confirmed"].tail(60)-datewise["Recovered"].tail(60)-datewise["Deaths"].tail(60))
# plt.title("Distributions plot for Active Cases")
# plt.xticks(rotation=45)
# plt.show()

def get_plot1():
    plt.switch_backend('AGG')
    plt.figure(figsize=(13,6))
    sns.barplot(x=datewise.index.date[-30:],y=datewise["Confirmed"].tail(30)-datewise["Recovered"].tail(30)-datewise["Deaths"].tail(30))
    plt.title("Distributions plot for Active Cases")
    plt.xticks(rotation=90)
    plt.xlabel('kanha xlable')
    plt.ylabel('kanha ylable')
    plt.tight_layout()
    graph = get_graph() 
    return graph

# Distribution plot  for Closed Cases
# plt.figure(figsize=(16,5))
# sns.barplot(x=datewise.index.date[-60:],y=datewise["Recovered"].tail(60)+datewise["Deaths"].tail(60))
# plt.title("Distribution plot  for Closed Cases")
# plt.xticks(rotation=90)
# plt.show()

def get_plot2():
    plt.switch_backend('AGG')
    plt.figure(figsize=(16,5))
    sns.barplot(x=datewise.index.date[-30:],y=datewise["Recovered"].tail(30)+datewise["Deaths"].tail(30))
    plt.title("Distribution plot  for Closed Cases")
    plt.xticks(rotation=90)
    plt.xlabel('kanha xlable')
    plt.ylabel('kanha ylable')
    plt.tight_layout()
    graph = get_graph() 
    return graph


# weak of the year data analysis
datewise["WeekofYear"] = datewise.index.isocalendar().week
week_num = []
weekwise_confirmed = []
weekwise_recovered = []
weekwise_deaths = []
w = 1
for i in list(datewise["WeekofYear"].unique()):
    weekwise_confirmed.append(datewise[datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])
    weekwise_recovered.append(datewise[datewise["WeekofYear"]==i]["Recovered"].iloc[-1])
    weekwise_deaths.append(datewise[datewise["WeekofYear"]==i]["Deaths"].iloc[-1])
    week_num.append(w)
    w=w+1

# Weekly Progress of different types of cases   
# plt.figure(figsize=(8,5))
# plt.plot(week_num,weekwise_confirmed,linewidth=3)
# plt.plot(week_num,weekwise_recovered,linewidth =3)
# plt.plot(week_num,weekwise_deaths,linewidth = 3)
# plt.xlabel("WeekNumber")
# plt.ylabel("Number of cases")
# plt.title("Weekly Progress of different types of cases")
# plt.show()

def get_plot3():
    plt.switch_backend('AGG')
    plt.figure(figsize=(8,5))
    plt.plot(week_num,weekwise_confirmed,linewidth=3)
    plt.plot(week_num,weekwise_recovered,linewidth =3)
    plt.plot(week_num,weekwise_deaths,linewidth = 3)
    plt.xlabel("WeekNumber")
    plt.ylabel("Number of cases")
    plt.title("Weekly Progress of different types of cases")
    plt.tight_layout()
    graph = get_graph() 
    return graph

# all are print
# x=print("Average increase in number of Confirmed cases everyday:",np.round(datewise["Confirmed"].diff().fillna(0).mean()))
# y=print("Average increase in number of Recovered cases everyday:",np.round(datewise["Recovered"].diff().fillna(0).mean()))
# z=print("Average increase in number of Death cases everyday:",np.round(datewise["Deaths"].diff().fillna(0).mean()))


# Daily increase in number cases in graph
# plt.figure(figsize=(15,6))
# plt.plot(datewise["Confirmed"][-10:].diff().fillna(0),label="Daily increase in confirmed cases",linewidth=5)
# plt.plot(datewise["Recovered"][-10:].diff().fillna(0),label="Daily increase in recovered cases",linewidth=5)
# plt.plot(datewise["Deaths"][-10:].diff().fillna(0),label="Daily inscrease in death cases",linewidth=5)
# plt.xlabel("Timestamp")
# plt.ylabel("Daily increase")
# plt.title("Daily increase")
# plt.legend()
# plt.xticks(rotation=90)
# plt.show()



# Daily increase graph
def get_plot4():
    plt.switch_backend('AGG')
    plt.figure(figsize=(15,6))
    plt.plot(datewise["Confirmed"][-10:].diff().fillna(0),label="Daily increase in confirmed cases",linewidth=5)
    plt.plot(datewise["Recovered"][-10:].diff().fillna(0),label="Daily increase in recovered cases",linewidth=5)
    plt.plot(datewise["Deaths"][-10:].diff().fillna(0),label="Daily inscrease in death cases",linewidth=5)
    plt.xlabel("Timestamp")
    plt.ylabel("Daily increase")
    plt.title("Daily increase")
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    graph = get_graph() 
    return graph


# #Country wise analysis
# #Calculating Country wise Mortality rate
countrywise= covid[covid["ObservationDate"]==covid["ObservationDate"].max()].groupby(["Country/Region"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"}).sort_values(["Confirmed"],ascending=False)
countrywise["Mortality"]=(countrywise["Deaths"]/countrywise["Recovered"])*100
countrywise["Recovered"]=(countrywise["Recovered"]/countrywise["Confirmed"])*100


# Top 15 countries as per number of confirmed cases
# fig,(ax1,ax2)=plt.subplots(1,2,figsize=(25,10))
# top_15confirmed = countrywise.sort_values(["Confirmed"],ascending=False).head(15)
# top_15deaths = countrywise.sort_values(["Deaths"],ascending=False).head(15)
# sns.barplot(x=top_15confirmed["Confirmed"],y=top_15confirmed.index,ax=ax1)
# ax1.set_title("Top 15 countries as per number of confirmed cases")
# sns.barplot(x=top_15deaths["Deaths"],y=top_15deaths.index,ax=ax2)
# ax2.set_title("Top 15 countries as per number of death cases")
# plt.show()


def get_plot5():
    plt.switch_backend('AGG')
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(25,10))
    top_15confirmed = countrywise.sort_values(["Confirmed"],ascending=False).head(15)
    top_15deaths = countrywise.sort_values(["Deaths"],ascending=False).head(15)
    sns.barplot(x=top_15confirmed["Confirmed"],y=top_15confirmed.index,ax=ax1)
    ax1.set_title("Top 15 countries as per number of confirmed cases")
    sns.barplot(x=top_15deaths["Deaths"],y=top_15deaths.index,ax=ax2)
    ax1.set_title("Top 15 countries as per number of death cases")
    plt.tight_layout()
    graph = get_graph() 
    return graph


#print this 
#Data Anlaysis for India
india_data = covid[covid["Country/Region"]=="India"]
datewise_india = india_data.groupby(["ObservationDate"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
# A = print(datewise_india.iloc[-1])
# B = print("Total Active Cases",datewise_india["Confirmed"].iloc[-1]-datewise_india["Recovered"].iloc[-1]-datewise_india["Deaths"].iloc[-1])
# C = print("Total Closed Cases",datewise_india["Recovered"].iloc[-1]+datewise_india["Deaths"].iloc[-1])


# not print
#Data Anlaysis for US
# us_data = covid[covid["Country/Region"]=="US"]
# datewise_us = us_data.groupby(["ObservationDate"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
# print(datewise_us.iloc[-1])
# print("Total Active Cases",datewise_us["Confirmed"].iloc[-1]-datewise_us["Recovered"].iloc[-1]-datewise_us["Deaths"].iloc[-1])
# print("Total Closed Cases",datewise_us["Recovered"].iloc[-1]+datewise_us["Deaths"].iloc[-1])


datewise_india["WeekofYear"] = datewise_india.index.isocalendar().week
week_num_india = []
india_weekwise_confirmed = []
india_weekwise_recovered = []
india_weekwise_deaths = []
w = 1
for i in list(datewise_india["WeekofYear"].unique()):
    india_weekwise_confirmed.append(datewise_india[datewise_india["WeekofYear"]==i]["Confirmed"].iloc[-1])
    india_weekwise_recovered.append(datewise_india[datewise_india["WeekofYear"]==i]["Recovered"].iloc[-1])
    india_weekwise_deaths.append(datewise_india[datewise_india["WeekofYear"]==i]["Deaths"].iloc[-1])
    week_num_india.append(w)
    w=w+1


# Weekly Progress of different types of cases for india    
# plt.figure(figsize=(8,5))
# plt.plot(week_num_india,india_weekwise_confirmed,linewidth=3)
# plt.plot(week_num_india,india_weekwise_recovered,linewidth =3)
# plt.plot(week_num_india,india_weekwise_deaths,linewidth = 3)
# plt.xlabel("WeekNumber")
# plt.ylabel("Number of cases")
# plt.title("Weekly Progress of different types of cases")
# plt.show()


# Weekly Progress of different types of cases for india
def get_plot6():
    plt.switch_backend('AGG')
    plt.figure(figsize=(8,5))
    plt.plot(week_num_india,india_weekwise_confirmed,linewidth=3)
    plt.plot(week_num_india,india_weekwise_recovered,linewidth =3)
    plt.plot(week_num_india,india_weekwise_deaths,linewidth = 3)
    plt.xlabel("WeekNumber")
    plt.ylabel("Number of cases")
    plt.title("Weekly Progress of different types of cases for india")
    plt.tight_layout()
    graph = get_graph() 
    return graph

# history of the covid cases
max_ind = datewise_india["Confirmed"].max()
china_data = covid[covid["Country/Region"]=="Mainland China"]
Italy_data = covid[covid["Country/Region"]=="Italy"]
US_data = covid[covid["Country/Region"]=="US"]
spain_data = covid[covid["Country/Region"]=="Spain"]
datewise_china = china_data.groupby(["ObservationDate"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
datewise_Italy = Italy_data.groupby(["ObservationDate"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
datewise_US=US_data.groupby(["ObservationDate"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
datewise_Spain=spain_data.groupby(["ObservationDate"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
# print("It took",datewise_india[datewise_india["Confirmed"]>0].shape[0],"days in India to reach",max_ind,"Confirmed Cases")
# print("It took",datewise_Italy[(datewise_Italy["Confirmed"]>0)&(datewise_Italy["Confirmed"]<=max_ind)].shape[0],"days in Italy to reach number of Confirmed Cases")
# print("It took",datewise_US[(datewise_US["Confirmed"]>0)&(datewise_US["Confirmed"]<=max_ind)].shape[0],"days in US to reach number of Confirmed Cases")
# print("It took",datewise_Spain[(datewise_Spain["Confirmed"]>0)&(datewise_Spain["Confirmed"]<=max_ind)].shape[0],"days in Spain to reach number of Confirmed Cases")
# print("It took",datewise_china[(datewise_china["Confirmed"]>0)&(datewise_china["Confirmed"]<=max_ind)].shape[0],"days in China to reach number of Confirmed Cases")

#below code is used to test and train the data to machine
datewise["Days Since"]=datewise.index-datewise.index[0]
datewise["Days Since"] = datewise["Days Since"].dt.days
train_ml = datewise.iloc[:int(datewise.shape[0]*0.95)]
valid_ml = datewise.iloc[:int(datewise.shape[0]*0.95):]
model_scores=[]




# linear regresssion
lin_reg = LinearRegression(normalize=True)
svm = SVR(C=1,degree=5,kernel='poly',epsilon=0.001)
lin_reg.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))
svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,))


prediction_valid_lin_reg = lin_reg.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
prediction_valid_svm = svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))

# date prediction for next to month
# print this in a table
new_date = []
new_prediction_lr=[]
new_prediction_svm=[]
for i in range(1,60):
    new_date.append(datewise.index[-1]+timedelta(days=i))
    new_prediction_lr.append(lin_reg.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0][0])
    new_prediction_svm.append(svm.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0])
pd.set_option("display.float_format",lambda x: '%.f' % x)
model_predictions=pd.DataFrame(zip(new_date,new_prediction_lr,new_prediction_svm),columns = ["Dates","LR","SVR"])
# model_predictions.tail(10)


model_train=datewise.iloc[:int(datewise.shape[0]*0.85)]
valid=datewise.iloc[int(datewise.shape[0]*0.85):]

# holt prediction
holt=Holt(np.asarray(model_train["Confirmed"])).fit(smoothing_level=1.4,smoothing_trend=0.2)
y_pred = valid.copy()
y_pred["Holt"]=holt.forecast(len(valid))


# print in a table
holt_new_date=[]
holt_new_prediction=[]
for i in range(1,60):
   holt_new_date.append(datewise.index+timedelta(days=i))
   holt_new_prediction.append(holt.forecast((len(valid)+i))[-1])

model_predictions["Holts Linear Model Prediction"]=holt_new_prediction
# model_predictions.tail(10) 

                           



# graph for holt prediction
# plt.figure(figsize=(15,5))
# sns.barplot(x=new_date,y=holt_new_prediction)
# plt.title("forecast plot for Active Cases")
# plt.xticks(rotation=90)
# plt.show()
def get_plot7():
    plt.switch_backend('AGG')
    plt.figure(figsize=(15,5))
    sns.barplot(x=new_date,y=holt_new_prediction)
    plt.title("forecast plot for Active Cases")
    plt.xticks(rotation=90)
    plt.tight_layout()
    graph = get_graph() 
    return graph


# prediction using lineaer regression 
# plt.figure(figsize=(15,5))
# sns.barplot(x=new_date,y=new_prediction_lr)
# plt.title("prediction plot for Active Cases")
# plt.xticks(rotation=90)
# plt.show()
def get_plot8():
    plt.switch_backend('AGG')
    plt.figure(figsize=(15,5))
    sns.barplot(x=new_date,y=new_prediction_lr)
    plt.title("prediction plot for Active Cases")
    plt.xticks(rotation=90)
    plt.tight_layout()
    graph = get_graph() 
    return graph


# prediction using SVM
# plt.figure(figsize=(15,5))
# sns.barplot(x=new_date,y=new_prediction_svm )
# plt.title("prediction plot for Active Cases")
# plt.xticks(rotation=90)
# plt.show()
def get_plot9():
    plt.switch_backend('AGG')
    plt.figure(figsize=(15,5))
    sns.barplot(x=new_date,y=new_prediction_svm )
    plt.title("prediction plot for Active Cases")
    plt.xticks(rotation=90)
    plt.tight_layout()
    graph = get_graph() 
    return graph




#fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4)
#sns.barplot(x= new_date,y=pd.Series(new_prediction_lr).diff().fillna(0),ax=ax1)
#sns.barplot(x=new_date,y=pd.Series(new_prediction_lr).diff().fillna(0),ax=ax1)
#sns.barplot(x=new_date,y=pd.Series(new_prediction_svm).diff().fillna(0),ax=ax2)
#sns.barplot(x=new_date,y=pd.Series(holt_new_prediction).diff().fillna(0),ax=ax3)
#sns.barplot(x=new_date,y=new_prediction_lr)
#sns.barplot(x=new_date,y=new_prediction_svm) 
#plt.plot(new_prediction_lr,linewidth =3)
#plt.plot(new_prediction_svm ,linewidth = 3)
#plt.xlabel("WeekNumber")
#plt.ylabel("Number of cases")
#plt.title("Weekly Progress of different types of cases")
#plt.show()  




                          

