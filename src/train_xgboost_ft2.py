# Importing Libraries
import numpy as np 
import pandas as pd 
import pickle
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna

# Creating DataFrames and removing index created during feature_engineering export
X_df = pd.read_csv("/workspace/anomaly_lte/data/x_train_processed_ft2.csv")
Y_df = pd.read_csv("/workspace/anomaly_lte/data/Y_train.csv")

X_df = X_df.iloc[:,1:]
Y_df = Y_df.iloc[:,1:]

# Creating Stratified K-Fold
RANDOM_SEED = 53
stratified_k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

# Creating Objective Function for Optuna
def xgboost_objective_function(trial):
    _n_estimators = trial.suggest_int("n_estimators", 50, 1000)
    #_early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 2, 50)
    _max_depth = trial.suggest_int("max_depth", 5, 500)
    _learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.1)
    _subsample = trial.suggest_float("subsample", 0, 1)
    _colsample_bytree = trial.suggest_float("colsample_bytree", 0, 1)
    _colsample_bylevel = trial.suggest_float("colsample_bylevel", 0, 1)

    xgboost_classifier = XGBClassifier(
        n_estimators=_n_estimators,
        #early_stopping_rounds=_early_stopping_rounds,
        max_depth=_max_depth,
        learning_rate=_learning_rate,
        subsample=_subsample,
        colsample_bytree=_colsample_bytree,
        colsample_bylevel=_colsample_bylevel,
        random_state=RANDOM_SEED
    )

    scores = cross_val_score(
        xgboost_classifier, X_df, Y_df, cv=stratified_k_fold, scoring='f1'
    )

    return scores.mean()

# Creating study object for Optuna
study = optuna.create_study(direction="maximize")

# Optimizing the study
study.optimize(xgboost_objective_function, n_trials=5)

# Fitting best model
best_parameters = study.best_params

xgboost_ft2 = XGBClassifier(random_state=RANDOM_SEED, **best_parameters)
xgboost_ft2.fit(X_df, Y_df)

# Exporting model as pickle files
with open('models/train_xgboost_ft2.pkl', 'wb') as model_file:
    pickle.dump(xgboost_ft2, model_file)
