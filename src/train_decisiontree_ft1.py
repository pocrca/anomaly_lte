# Importing Libraries
import numpy as np 
import pandas as pd 
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Creating DataFrames and removing index created during feature_engineering export
X_df = pd.read_csv("/workspace/anomaly_lte/data/x_train_processed_ft1.csv")
Y_df = pd.read_csv("/workspace/anomaly_lte/data/Y_train.csv")

X_df = X_df.iloc[:,1:]
Y_df = Y_df.iloc[:,1:]

# Creating DecisionTree Classifiers
for fold in range(1,6):
    globals()[f"dtree_ft1_fold{fold}"] = DecisionTreeClassifier(random_state=53)

# Creating Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 53)

# Fitting Models
Fold = 0

for train_index, test_index in skf.split(X_df, Y_df):
    Fold += 1
    X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index],
    Y_train, Y_test = Y_df.iloc[train_index], Y_df.iloc[test_index],
    globals()[f"dtree_ft1_fold{Fold}"].fit(X_train, Y_train)
    with open(f'models/train_dtree_ft1.pkl', 'wb') as model_file:
        pickle.dump(dtree_ft1_fold1, model_file)
        pickle.dump(dtree_ft1_fold2, model_file)
        pickle.dump(dtree_ft1_fold3, model_file)
        pickle.dump(dtree_ft1_fold4, model_file)
        pickle.dump(dtree_ft1_fold5, model_file)

