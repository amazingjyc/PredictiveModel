import matplotlib
import sklearn
import pandas as pd
import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm

import re
import csv

def Build_Test_Set(features = ["Diff_Close","GSA_polarity"]):
    try:
        data_df = pd.DataFrame.from_csv("key_stats.csv")
        data_df = data_df[30:49]
        X = np.array(data_df[features].values)
    
    except Exception as e:
        print(str(e))
              
    return X

def Build_Data_Set(features = ["Diff_Close","GSA_polarity"]):
    try:
        data_df = pd.DataFrame.from_csv("key_stats.csv")
        data_df = data_df[1:29]
        X = np.array(data_df[features].values)
    
        y = (data_df["Pred_Label"]
            .replace("Bearish",0)
           .replace("Bullish",1)
          .values.tolist())
        
    except Exception as e:
        print(str(e))
              
    return X,y


ticker_list = []

def Analysis():
    try:
        X, y = Build_Data_Set()

        #Training..
        clf = svm.SVC(kernel="linear", C= 1.0)
        clf.fit(X,y)       
        w = clf.coef_[0]
        a = -w[0] / w[1]     
        xx = np.linspace(min(X[:, 0]), max(X[:, 0]))
        yy = a * xx - clf.intercept_[0] / w[1]
        h0 = plt.plot(xx,yy, "k-", label="Linear Support Vector Classification.")
        plt.scatter(X[:, 0],X[:, 1],c=y)
        plt.ylabel("GSA_polarity")
        plt.xlabel("Diff_Close")
        plt.legend()
        plt.show()
        
        #Prediction..
        X = Build_Test_Set()
        print(clf.predict(X))         
        #w = clf.coef_[0]
        #print(w)
        
        #a = -w[0] / w[1]
        
        #xx = np.linspace(min(X[:, 0]), max(X[:, 0]))
        #yy = a * xx - clf.intercept_[0] / w[1]
        
        #h0 = plt.plot(xx, yy, 'k-', label="non weighted div")
        
        #plt.scatter(X[:, 0], X[:, 1], c = y)
        #plt.legend()
        #plt.show()        
        
    except Exception as e:
        print(str(e))    

try:
    df = pd.DataFrame(columns = ['Ticker','Date','GSA_polarity','Diff_Close','Pred_Label'])
except:
    pass

#Building the Dataset..
with open('Dataset.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        
        try:
            ticker_list.append(row[0])
            date_stamp = datetime.strptime(row[1],'%d/%m/%Y')
            
            if row[2] == "neutral":
                sentiment = 0
            elif row[2] == "positive":
                sentiment = 1
            else:
                sentiment = -1
        
            df = df.append({'Ticker':row[0],'Date':date_stamp,'GSA_polarity':sentiment,'Diff_Close':float(row[3]),'Pred_Label':row[4]
                            ,}, ignore_index = True)

        except Exception as e:
            print(str(e))
          
#Plotting Sentiment Scores/ Close Values..
#for each_ticker in ticker_list:
    #try:
        #plot_df = df[(df['Ticker']==each_ticker)]
        #plot_df = plot_df.set_index(['Date'])
        
        #if plot_df['Status'][-1] == 'Underperformed':
            #color = 'r'
        #else:
            #color = 'g'        
        
        ##Change Sentiment Scores to 'Close' to compare..
        #plot_df['Sentiment Scores'].plot(label=each_ticker)
        ##plt.legend()

    #except Exception as e:
        #print(str(e))
        #pass
    
#plt.show()

#Save the DataFrame..
try:
    save = ('C:\\Users\\jarre\\Desktop\\PredModel\\key_stats.csv')
    df.to_csv(save)
    Analysis()
except Exception as e:
    print(str(e))