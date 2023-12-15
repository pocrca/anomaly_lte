# Importing Libraries
import pickle
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer

# Creating Dataframes
X_test = pd.read_csv("/workspace/anomaly_lte/data/X_test.csv")
Y_test = pd.read_csv("/workspace/anomaly_lte/data/Y_test.csv")
X_train = pd.read_csv("/workspace/anomaly_lte/data/X_train.csv")
Y_train = pd.read_csv("/workspace/anomaly_lte/data/Y_train.csv")

# Turning time into a datetime type
X_train['Time'] = pd.to_datetime(X_train['Time'], format = '%H:%M')
X_test['Time'] = pd.to_datetime(X_test['Time'], format = '%H:%M')

# Creating new columns for meanUE_UL_encoded and meanUE_DL_encoded (encoding will be done later)
X_train['meanUE_UL_encoded'] = X_train['meanUE_UL']
X_train['meanUE_DL_encoded'] = X_train['meanUE_DL']
X_test['meanUE_UL_encoded'] = X_test['meanUE_UL']
X_test['meanUE_DL_encoded'] = X_test['meanUE_DL']

# FunctionTransformer for custom One Hot Encoder for variables MeanUE_UL and MeanUE_DL
def zero_encoder(x):
    return (x == 0).astype(int)

Zero_Encoder = FunctionTransformer(zero_encoder)

# FunctionTransformer for encoding time as hours
def time_encoder(x):
    return np.array(x.iloc[:,0].dt.hour)[:, np.newaxis]

Time_Encoder = FunctionTransformer(time_encoder)

# FunctionTransformer and ColumnTransformer for log transformation 
def log_transformer(x):
    return np.log(x+10**-10) #Constant added to prevent log 0

Log_Transformer = FunctionTransformer(log_transformer)

# Pipeline for log transformation and standard scaler
Pipe = Pipeline(steps = [
    ('log', Log_Transformer),
    ('scale', StandardScaler())
])

# Fitting ColumnTransformer for set 1 (no log transformation)
preprocessor_set_1 = ColumnTransformer(
    transformers=[
        ('encode cell name', OneHotEncoder(), ['CellName']),
        ('encode time', Time_Encoder, ['Time']),
        ('encode zero', Zero_Encoder, ['meanUE_UL_encoded', 'meanUE_DL_encoded']),
        ('scale', StandardScaler(), ['PRBUsageUL', 'PRBUsageDL', 'meanThr_DL', 'meanThr_UL', 'maxThr_DL', 'maxThr_UL', 'meanUE_UL', 
        'meanUE_DL', 'maxUE_UL', 'maxUE_DL', 'maxUE_UL+DL'])
    ],
    remainder = 'passthrough'
)

preprocessor_set_1.fit(X_train)

# Fitting ColumnTranformer for set 2 (with log transformation)
preprocessor_set_2 = ColumnTransformer(
    transformers=[
        ('encode cell name', OneHotEncoder(), ['CellName']),
        ('encode time', Time_Encoder, ['Time']),
        ('encode zero', Zero_Encoder, ['meanUE_UL_encoded', 'meanUE_DL_encoded']),
        ('log and scale', Pipe, ['PRBUsageUL', 'PRBUsageDL', 'meanThr_DL', 'meanThr_UL', 'maxThr_DL', 'maxThr_UL', 'meanUE_UL', 'meanUE_DL']),
        ('scale', StandardScaler(), ['maxUE_UL', 'maxUE_DL', 'maxUE_UL+DL'])
    ],
    remainder = 'passthrough'
)

preprocessor_set_2.fit(X_train)

# Transforming Data
x_train_processed_ft1 = preprocessor_set_1.transform(X_train)
x_test_processed_ft1 = preprocessor_set_1.transform(X_test)
x_train_processed_ft2 = preprocessor_set_2.transform(X_train)
x_test_processed_ft2 = preprocessor_set_2.transform(X_test)

# Creating new DataFrames
x_train_processed_ft1 = pd.DataFrame(x_train_processed_ft1)
x_test_processed_ft1 = pd.DataFrame(x_test_processed_ft1)
x_train_processed_ft2 = pd.DataFrame(x_train_processed_ft2)
x_test_processed_ft2 = pd.DataFrame(x_test_processed_ft2)

# Creating new column names
Cells = []

for i in X_train.CellName:
    if i not in Cells:
        Cells = np.append(Cells, i) # Obtains unique cells

Columns = []

for i in Cells:
    Columns.append(f"Cell ID: {i}") # Creates column names for CellName One hot Encoding columns

Columns = Columns + ['Time', 'Mean UE devices encoded (uplink)', 'Mean UE devices encoded (downlink)', 'Percentage of PRB usage (uplink)', 'Percentage of PRB usage (downlink)', 'Mean carried traffic (downlink)', 'Mean carried traffic (uplink)', 'Max carried traffic (downlink)', 'Max carried traffic (uplink)', 'Mean UE devices (downlink)', 'Mean UE devices (uplink)',
'Max UE devices (downlink)', 'Max UE devices (uplink)', 'Max UE devices (uplink and downlink)', 'Outdated_Index'] 

x_train_processed_ft1.columns = Columns
x_test_processed_ft1.columns = Columns
x_train_processed_ft2.columns = Columns
x_test_processed_ft2.columns = Columns

# Dropping index created during Data Preperation
x_train_processed_ft1.drop(x_train_processed_ft1.columns[[-1]], axis = 1, inplace = True)
x_test_processed_ft1.drop(x_test_processed_ft1.columns[[-1]], axis = 1, inplace = True)
x_train_processed_ft2.drop(x_train_processed_ft2.columns[[-1]], axis = 1, inplace = True)
x_test_processed_ft2.drop(x_test_processed_ft2.columns[[-1]], axis = 1, inplace = True)

# Removing maxUE_UL, maxUE_UL+DL, Time (insignificant features)
x_train_processed_ft1 = x_train_processed_ft1.drop(["Max UE Devices (uplink)", "Max UE Devices (uplink and downlink)", "Time"], axis=1)
x_test_processed_ft1 = x_test_processed_ft1.drop(["Max UE Devices (uplink)", "Max UE Devices (uplink and downlink)", "Time"], axis=1)
x_train_processed_ft2 = x_train_processed_ft2.drop(["Max UE Devices (uplink)", "Max UE Devices (uplink and downlink)", "Time"], axis=1)
x_test_processed_ft2 = x_test_processed_ft2.drop(["Max UE Devices (uplink)", "Max UE Devices (uplink and downlink)", "Time"], axis=1)

# Exporting to csv files
x_train_processed_ft1.to_csv('data/x_train_processed_ft1.csv')
x_test_processed_ft1.to_csv('data/x_test_processed_ft1.csv')
x_train_processed_ft2.to_csv('data/x_train_processed_ft2.csv')
x_test_processed_ft2.to_csv('data/x_test_processed_ft2.csv')

# Exporting preprocessors 
with open('preprocessors/preprocessor_set_1', 'wb') as file:
    pickle.dump(preprocessor_set_1, file)
with open('preprocessors/preprocessor_set_2', 'wb') as file:
    pickle.dump(preprocessor_set_2, file)