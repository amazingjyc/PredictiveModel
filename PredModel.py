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

def Build_Data_Set(features = ["Magnitude",
                               "Sentiment Scores"]):
    try:
        data_df = pd.DataFrame.from_csv("key_stats.csv")
        data_df = data_df[1:31]
        X = np.array(data_df[features].values)
    
        y = (data_df["Label"]
            .replace("Bearish",0)
           .replace("Bullish",1)
          .values.tolist())
        
        # 0 - Bearish and 1 - Bullish
    except Exception as e:
        print(str(e))
              
    return X,y


ticker_list = []

def Analysis():
    try:
        X, y = Build_Data_Set()

        clf = svm.SVC(kernel="linear", C= 1.0)
        clf.fit(X,y)
        #Prediction, reuses the training set..
        print(clf.predict(X[0:7]))        
    
        w = clf.coef_[0]
        a = -w[0] / w[1]     
        xx = np.linspace(min(X[:, 0]), max(X[:, 0]))
        yy = a * xx - clf.intercept_[0] / w[1]

        h0 = plt.plot(xx,yy, "k-", label="Linear Support Vector Classification.")

        plt.scatter(X[:, 0],X[:, 1],c=y)
        plt.ylabel("Sentiment Scores")
        plt.xlabel("Magnitude")
        plt.legend()

        plt.show()
        
    except Exception as e:
        print(str(e))    

#try:
    #df = pd.DataFrame(columns = ['Ticker','Date','Sentiment Scores','Magnitude','Close', 'Difference_Close','Status'])
#except:
    #pass

#previous_price = 0.0

#Building the Dataset..
#with open('ScikitLearn_DataSet.csv') as csvfile:
    #readCSV = csv.reader(csvfile, delimiter=',')
    #for row in readCSV:
        
        #try:
            #ticker_list.append(row[0])
            #if previous_price > float(row[4]):
                #status = "Underperformed"
            #else:
                #status = "Outperformed" 
            #difference = float(row[4])-previous_price
            #previous_price = float(row[4])
           
            #date_stamp = datetime.strptime(row[1],'%m/%d/%Y')
        
            #df = df.append({'Ticker':row[0],'Date':date_stamp,'Sentiment Scores':float(row[2]),'Magnitude':float(row[3]),'Close':float(row[4]),
                        #'Difference_Close':difference,'Status':status,}, ignore_index = True)
            #previous_price = float(row[4])
        #except Exception as e:
            #print(str(e))
          
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
    #save = ('C:\\Users\\jarre\\Desktop\\PredModel\\key_stats.csv')
    #df.to_csv(save)
    Analysis()
except Exception as e:
    print(str(e))