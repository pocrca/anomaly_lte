# Importing Libraries
import numpy as np 
import pandas as pd 
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna

# Creating DataFrames and removing index created during feature_engineering export
X_df = pd.read_csv("/workspace/anomaly_lte/data/x_train_processed_ft1.csv")
Y_df = pd.read_csv("/workspace/anomaly_lte/data/Y_train.csv")

X_df = X_df.iloc[:,1:]
Y_df = Y_df.iloc[:,1:]

# Creating Stratified K-Fold
RANDOM_SEED = 53
stratified_k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

# Creating Objective Function for Optuna
def decision_tree_objective_function(trial):
    _max_depth = trial.suggest_int("max_depth", 5, 500)
    _min_samples_split = trial.suggest_int("min_samples_split", 2, 100)
    _min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 100)
    _max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 2000)
    _ccp_alpha = trial.suggest_float("ccp_alpha", 0, 0.01)

    decision_tree_classifier = DecisionTreeClassifier(
        max_depth=_max_depth,
        min_samples_split=_min_samples_split,
        min_samples_leaf=_min_samples_leaf,
        max_leaf_nodes=_max_leaf_nodes,
        ccp_alpha=_ccp_alpha,
        random_state=RANDOM_SEED
    )

    scores = cross_val_score(
        decision_tree_classifier, X_df, Y_df, cv=stratified_k_fold, scoring='f1'
    )

    return scores.mean()

# Creating study object for Optuna
study = optuna.create_study(direction="maximize")
print("y")
# Optimizing the study
study.optimize(decision_tree_objective_function, n_trials=100) # n_trials set low temporarily
print("y2")
# Fitting best model
best_parameters = study.best_params

dtree_ft1 = DecisionTreeClassifier(random_state=RANDOM_SEED, **best_parameters)
dtree_ft1.fit(X_df, Y_df)

# Exporting model as pickle files
with open('models/train_dtree_ft1.pkl', 'wb') as model_file:
    pickle.dump(dtree_ft1, model_file)



