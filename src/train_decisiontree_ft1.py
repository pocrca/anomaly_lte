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

# Creating classifier
classifier = DecisionTreeClassifier(random_state=53)

# Creating Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 53)

# Fitting Model
scores = []
AUCs = []
Fold_number = 0
Mean_cm = [[0,0],[0,0]]

for train_index, test_index in skf.split(X_df, Y_df):
    X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index],
    Y_train, Y_test = Y_df.iloc[train_index], Y_df.iloc[test_index],
    classifier.fit(X_train, Y_train)
    Predicted = classifier.predict(X_test)

    # Getting Scores
    score = classifier.score(X_test,Y_test)
    scores.append(score)

    # Getting confusion matrix
    cm = confusion_matrix(Y_test, Predicted)
    Mean_cm = Mean_cm + cm 

    # Getting ROC Curve
    fpr, tpr, thresholds = roc_curve(Y_test, Predicted)

    # Getting AUC values
    AUC = auc(fpr, tpr)
    AUCs.append(AUC)

    # Plotting ROC Curve
    Fold_number += 1
    plt.plot(fpr, tpr, label = f"ROC Curve | AUC = {round(AUC,3)}, Score = {round(score,3)}")
    plt.plot([0,1], [0,1], linestyle = '--', label = "Baseline")
    plt.title(f"Fold {Fold_number}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid()
    plt.legend()
    plt.show()

# Printing overall evaluation metrics
print(f"Mean Score: {np.mean(scores)}")
print(f"Mean AUC: {np.mean(AUCs)}")
print(f"Mean Confusion Matrix: \n{Mean_cm/5}")

# Saving Models as pickle files
with open('train_dtree_ft1.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)