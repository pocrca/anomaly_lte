# Importing libraries
import streamlit as st
import numpy as np 
import pandas as pd 
import pickle
import matplotlib.pyplot as plt
import shap
from streamlit_shap import st_shap
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
import joblib

# Importing decision tree models
with open('models/train_dtree_ft2.pkl', 'rb') as model_file:
    dtree = pickle.load(model_file)
with open('models/train_xgboost_ft2.pkl', 'rb') as model_file:
    xgboost = pickle.load(model_file)

# Creating Dataframes
X_train = pd.read_csv("/workspace/anomaly_lte/data/X_train.csv")

# Turning time into a datetime type
X_train['Time'] = pd.to_datetime(X_train['Time'], format = '%H:%M')


# Creating new columns for meanUE_UL_encoded and meanUE_DL_encoded (encoding will be done later)
X_train['meanUE_UL_encoded'] = X_train['meanUE_UL']
X_train['meanUE_DL_encoded'] = X_train['meanUE_DL']


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

# Fitting ColumnTranformer for set 2 (with log transformation)
preprocessor = ColumnTransformer(
    transformers=[
        ('encode cell name', OneHotEncoder(), ['CellName']),
        ('encode time', Time_Encoder, ['Time']),
        ('encode zero', Zero_Encoder, ['meanUE_UL_encoded', 'meanUE_DL_encoded']),
        ('log and scale', Pipe, ['PRBUsageUL', 'PRBUsageDL', 'meanThr_DL', 'meanThr_UL', 'maxThr_DL', 'maxThr_UL', 'meanUE_UL', 'meanUE_DL']),
        ('scale', StandardScaler(), ['maxUE_UL', 'maxUE_DL', 'maxUE_UL+DL'])
    ],
    remainder = 'passthrough'
)

preprocessor.fit(X_train)

# Streamlit configuration
st.set_page_config(layout="wide")

# Description and title
st.title('Anomaly Detection in LTE Network')

with st.expander('About The Project'):
    st.write("Traditionally, the design of a cellular network focuses on the optimization of resources that guarantees a smooth operation even during peak hours (i.e. periods with higher traffic load). However, this implies that cells are most of the time overprovisioned of radio resources. Next generation cellular networks ask for a dynamic management and configuration in order to adapt to the varying user demands to utilize frequency resources more efficiently. If the network operator is capable of anticipating to variations in users’ traffic demands, a more efficient management of scarce network resources would be possible.")
    st.write("As such, the project aims to:")
    st.markdown("* Explore the possibilities of ML to detect abnormal behaviors in the utilization of the network that would motivate a change in the configuration of the base station.")
    st.markdown("* Analyze a [dataset](https://www.kaggle.com/competitions/anomaly-detection-in-4g-cellular-networks/overview) (public) of past traces of LTE activity and use it to train an ML model capable of classifying samples of current activity as:")
    st.markdown("    - (a) **Normal** Activity, therefore, no re-configuration or redistribution of resources is needed.")
    st.markdown("    - (b) **Unusual** activity, which differs from the behavior usually observed and should trigger a reconfiguration of the base station.")


with st.expander('Project Methodology'):
    st.image("MLP.JPG", caption="Machine Learning Pipeline", width = 500)
    st.write("The two classification models being used in this project are:")
    st.markdown("1. Decision Tree") 
    st.markdown("2. XGBoost (Extreme Gradient Boosting)")

# Inputting test data
st.subheader("Testing out the models")
uploaded_file = st.file_uploader("Input Raw Testing Data")

if uploaded_file is not None:
    X_df_raw = pd.read_csv(uploaded_file)
    
    # Adding Index
    X_df_raw.insert(0, 'Unnamed: 0', np.arange(0, 101, 1))

    # Turning time into a datetime type
    X_df_raw['Time'] = pd.to_datetime(X_df_raw['Time'], format = '%H:%M')

    # Creating new columns for meanUE_UL_encoded and meanUE_DL_encoded 
    X_df_raw['meanUE_UL_encoded'] = X_df_raw['meanUE_UL']
    X_df_raw['meanUE_DL_encoded'] = X_df_raw['meanUE_DL']

    # Feature engineering
    X_df = preprocessor.transform(X_df_raw)
    X_df = pd.DataFrame(X_df)

    # Creating new column names
    Columns = ['CellName_7VLTE', 'CellName_5ALTE', 'CellName_10ALTE', 'CellName_9BLTE', 'CellName_7BLTE', 'CellName_3BLTE', 
    'CellName_5BLTE', 'CellName_9ALTE', 'CellName_10CLTE', 'CellName_7ULTE', 'CellName_6ULTE', 'CellName_1BLTE', 
    'CellName_2ALTE', 'CellName_7ALTE', 'CellName_8ALTE', 'CellName_6ALTE', 'CellName_3CLTE', 'CellName_6CLTE', 
    'CellName_10BLTE', 'CellName_6WLTE', 'CellName_8CLTE', 'CellName_3ALTE', 'CellName_4CLTE', 'CellName_4BLTE', 
    'CellName_7WLTE', 'CellName_6VLTE', 'CellName_1ALTE', 'CellName_7CLTE', 'CellName_4ALTE', 'CellName_1CLTE', 
    'CellName_8BLTE', 'CellName_5CLTE', 'CellName_6BLTE', 'Time', 'meanUE_UL_encoded', 'meanUE_DL_encoded', 'PRBUsageUL', 
    'PRBUsageDL', 'meanThr_DL', 'meanThr_UL', 'maxThr_DL', 'maxThr_UL', 'meanUE_DL', 'meanUE_UL', 'maxUE_DL', 'maxUE_UL', 
    'maxUE_UL+DL', 'Outdated Index']

    X_df.columns = Columns

    # Dropping index
    X_df.drop(X_df.columns[[-1]], axis = 1, inplace = True)

    # Setting up SHAP explainer
    explainer_dtree = shap.TreeExplainer(dtree)
    explainer_xgboost = shap.TreeExplainer(xgboost)
    shap_values_dtree = explainer_dtree.shap_values(X_df)
    shap_values_xgboost = explainer_xgboost.shap_values(X_df)

else:
    st.info('☝️ Upload a CSV file first')

# Creating sidebar 
st.sidebar.header('Models and Evaluation Metrics')
selected_model = []
selected_evaluations = []

if uploaded_file is not None:
    selected_model = st.sidebar.selectbox("Selected Model:", ('Decision Tree', "XGBoost"))
    if selected_model is not None:
        selected_evaluations = st.sidebar.multiselect("Evaluation metrics:", ["SHAP Tree Explainer Bar Chart", "F1 Score (Temporary Placeholder)"])
    else:
        st.sidebar.info('☝️ Choose a model to evaluate')
else:
    st.sidebar.info('Input testing data first')

# Showing evaluation for SHAP
if ("SHAP Tree Explainer Bar Chart" in selected_evaluations) and (selected_model == 'Decision Tree'):
    st.markdown("**SHAP Tree Explainer Bar Chart:**")
    st_shap(shap.summary_plot(shap_values_dtree, X_df, plot_type='bar', class_names=['Normal', 'Anomalous']))
elif ("SHAP Tree Explainer Bar Chart" in selected_evaluations) and (selected_model == 'XGBoost'):
    st.markdown("**SHAP Tree Explainer Bar Chart:**")
    st_shap(shap.summary_plot(shap_values_xgboost, X_df, plot_type='bar', class_names=['Normal', 'Anomalous']))
else:
    pass