#Importing libraries
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

#Removing outlier
df = pd.read_csv("/workspace/anomaly_lte/data/train.csv", sep = ';')

df = df[df.meanUE_UL != np.max(df.meanUE_UL)]

#Splitting data
X = df[df.columns[0:13]]
Y = df[df.columns[13]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 53, test_size = 0.2, shuffle = True)

#Exporting dataframes to csv files
X_train.to_csv('X_train.csv')
X_test.to_csv('X_test.csv')
Y_train.to_csv('Y_train.csv')
Y_test.to_csv('Y_test.csv')