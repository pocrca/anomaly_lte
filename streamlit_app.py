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
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix, classification_report, precision_recall_curve, PrecisionRecallDisplay, average_precision_score

# Importing models
with open('models/train_dtree_ft2.pkl', 'rb') as model_file:
    dtree = pickle.load(model_file)
with open('models/train_xgboost_ft2.pkl', 'rb') as model_file:
    xgboost = pickle.load(model_file)

# Defining Feature Engineering functions 
def zero_encoder(x):
    return (x == 0).astype(int)
Zero_Encoder = FunctionTransformer(zero_encoder)

def time_encoder(x):
    return np.array(x.iloc[:,0].dt.hour)[:, np.newaxis]
Time_Encoder = FunctionTransformer(time_encoder)

def log_transformer(x):
    return np.log(x+10**-10) #Constant added to prevent log 0
Log_Transformer = FunctionTransformer(log_transformer)

# Importing preprocessor for feature engineering
with open('preprocessors/preprocessor_set_2', 'rb') as file:
    preprocessor = pickle.load(file)

# Streamlit configuration
st.set_page_config(layout="wide")

# Creating columns to centralize text
left_column, centre_column, right_column = st.columns([1,8,1])

# Description and title
with centre_column:
    st.title("Anomaly Detection in LTE Network")

    with st.expander('About The Project'):
        st.write("Next generation cellular networks ask for a dynamic management and configuration in order to adapt to the varying user demands to utilize frequency resources more efficiently. If the network operator is capable of anticipating to variations in users’ traffic demands, a more efficient management of scarce network resources would be possible.")
        st.write("As such, the project aims to:")
        st.markdown("* Explore the possibilities of ML to detect abnormal behaviors in the utilization of the network that would motivate a change in the configuration of the base station.")
        st.markdown("* Analyze a [dataset](https://www.kaggle.com/competitions/anomaly-detection-in-4g-cellular-networks/overview) (public) of past traces of LTE activity and use it to train an ML model capable of classifying samples of current activity as:")
        st.markdown("    - (a) **Normal** activity, therefore, no re-configuration or redistribution of resources is needed.")
        st.markdown("    - (b) **Unusual** activity, which differs from the behavior usually observed and should trigger a reconfiguration of the base station.")

    with st.expander('Project Methodology'):
        st.image("MLP.JPG", caption="Machine Learning Pipeline", width = 570)
        st.write("The two classification models being used in this project are:")
        st.markdown("1. Decision Tree") 
        st.markdown("2. XGBoost (Extreme Gradient Boosting)")

# Inputting test data
with centre_column:
    st.header("\n\n\n")
    st.header("Test out the models")
    uploaded_X_data = st.file_uploader("Input testing data (predictor features)")
    uploaded_Y_data = st.file_uploader("Input testing data (outcome feature / results)")

if uploaded_X_data is not None:
    X_df_raw = pd.read_csv(uploaded_X_data)
    
    # Adding Index
    X_df_raw.insert(0, 'Unnamed: 0', np.arange(0, 100, 1))

    # Turning time into a datetime type
    X_df_raw['Time'] = pd.to_datetime(X_df_raw['Time'], format = '%H:%M')

    # Creating new columns for meanUE_UL_encoded and meanUE_DL_encoded 
    X_df_raw['meanUE_UL_encoded'] = X_df_raw['meanUE_UL']
    X_df_raw['meanUE_DL_encoded'] = X_df_raw['meanUE_DL']

    # Feature engineering
    X_df = preprocessor.transform(X_df_raw)
    X_df = pd.DataFrame(X_df)

    # Creating new column names
    Columns = ['Cell ID: 7VLTE', 'Cell ID: 5ALTE', 'Cell ID: 10ALTE', 'Cell ID: 9BLTE', 'Cell ID: 7BLTE', 
        'Cell ID: 3BLTE', 'Cell ID: 5BLTE', 'Cell ID: 9ALTE', 'Cell ID: 10CLTE', 'Cell ID: 7ULTE', 
        'Cell ID: 6ULTE', 'Cell ID: 1BLTE', 'Cell ID: 2ALTE', 'Cell ID: 7ALTE', 'Cell ID: 8ALTE', 
        'Cell ID: 6ALTE', 'Cell ID: 3CLTE', 'Cell ID: 6CLTE', 'Cell ID: 10BLTE', 'Cell ID: 6WLTE', 
        'Cell ID: 8CLTE', 'Cell ID: 3ALTE', 'Cell ID: 4CLTE', 'Cell ID: 4BLTE', 'Cell ID: 7WLTE', 
        'Cell ID: 6VLTE', 'Cell ID: 1ALTE', 'Cell ID: 7CLTE', 'Cell ID: 4ALTE', 'Cell ID: 1CLTE', 
        'Cell ID: 8BLTE', 'Cell ID: 5CLTE', 'Cell ID: 6BLTE', 'Time', 'Mean UE Devices (uplink)', 
        'Mean UE Devices (downlink)', 'Percentage of PRB Usage (uplink)', 'Percentage of PRB Usage (downlink)',
        'Mean Carried Traffic (downlink)', 'Mean Carried Traffic (uplink)', 'Max Carried Traffic (downlink)', 
        'Max Carried Traffic (uplink)', 'Mean UE Devices (downlink)', 'Mean UE Devices (uplink)', 
        'Max UE Devices (downlink)', 'Max UE Devices (uplink)', 'Max UE Devices (uplink and downlink)', 
        'Outdated_Index']

    X_df.columns = Columns

    # Dropping index
    X_df.drop(X_df.columns[[-1]], axis = 1, inplace = True)

    # Removing maxUE_UL, maxUE_UL+DL, Time (insignificant features)
    X_df = X_df.drop(["Max UE Devices (uplink)", "Max UE Devices (uplink and downlink)", "Time"], axis=1)


if uploaded_Y_data is not None:
    Y_df = pd.read_csv(uploaded_Y_data)

# Creating sidebar for evaluation metrics
st.sidebar.title('Model Diagnostics')
selected_dtree = False
selected_xgboost = False
selected_shap = False
selected_score = False
selected_roc = False
selected_prd = False

if uploaded_X_data is not None and uploaded_Y_data is not None:
    st.sidebar.markdown("**Selected Model:**")
    selected_dtree = st.sidebar.checkbox("Decision Tree")
    selected_xgboost = st.sidebar.checkbox("XGBoost")

    if selected_dtree or selected_xgboost:
        st.sidebar.markdown("**Evaluation Metrics:**")
        selected_shap = st.sidebar.checkbox("SHAP Tree Explainer Bar Chart")
        selected_score = st.sidebar.checkbox("Predictions and Score")
        selected_roc = st.sidebar.checkbox("ROC (Receiver Operating Characteristic)")
        selected_prd = st.sidebar.checkbox("PRD (Precision Recall Display)")

    else:
        st.sidebar.info('☝️ Choose a model to evaluate')

elif uploaded_X_data is not None and uploaded_Y_data is None: # Function to just predict and show SHAP
    st.sidebar.markdown("**Selected Model:**")
    selected_dtree = st.sidebar.checkbox("Decision Tree")
    selected_xgboost = st.sidebar.checkbox("XGBoost")

    if selected_dtree or selected_xgboost:
        st.sidebar.markdown("**Evaluation Metrics:**")
        selected_shap = st.sidebar.checkbox("SHAP Tree Explainer Bar Chart")
        selected_predictions = st.sidebar.checkbox("Predictions")
        st.sidebar.info('🔒 Upload testing data results as well for other evaluation metrics')

else:
    st.sidebar.info('Input testing data first')
    centre_column.info('☝️ Upload CSV files')

# Header for model diagnostics
if selected_shap or selected_score or selected_roc or selected_prd:
    if (selected_dtree and selected_xgboost) and (selected_score or selected_shap) :
        st.text("")
        st.header("Model Evaluation:")

    else: 
        with centre_column:
            st.text("")
            st.header("Model Evaluation:")

# Showing evaluations for Predictions (evaluation option that only pops up when result dataset is not added)
def evaluations_predictions(model1, model2):
    if model2 is None:
        with centre_column:
            st.text("")
            st.subheader("Model Predictions")

            # Creating and showing predicted values and probabilities
            predicted_probability_anomalous = model1.predict_proba(X_df)[:,1]
            predicted_probability_normal = model1.predict_proba(X_df)[:,0]
            predicted_results = model1.predict(X_df)

            predicted_results = map(lambda x: "Normal" if x == 0 else "Anomaly", predicted_results)
            predicted_probability_normal = map(lambda x: f"{round(x*100,1)}%", predicted_probability_normal)
            predicted_probability_anomalous = map(lambda x: f"{round(x*100,1)}%", predicted_probability_anomalous)

            results = {
                "Predicted Behaviour": predicted_results,
                "Probability of being normal": predicted_probability_normal,
                "Probability of being anomalous": predicted_probability_anomalous
            }
        
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

    else: 
        st.text("")
        st.text("Model Predictions")

        # Creating two columns
        predictions_left_column, predictions_right_column = st.columns([1,1])

        with predictions_left_column:
            st.text("Decision Tree:")

            # Creating and showing predicted values and probabilities
            predicted_probability_anomalous = model1.predict_proba(X_df)[:,1]
            predicted_probability_normal = model1.predict_proba(X_df)[:,0]
            predicted_results = model1.predict(X_df)

            predicted_results = map(lambda x: "Normal" if x == 0 else "Anomaly", predicted_results)
            predicted_probability_normal = map(lambda x: f"{round(x*100,1)}%", predicted_probability_normal)
            predicted_probability_anomalous = map(lambda x: f"{round(x*100,1)}%", predicted_probability_anomalous)

            results = {
                "Predicted Behaviour": predicted_results,
                "Probability of being normal": predicted_probability_normal,
                "Probability of being anomalous": predicted_probability_anomalous
            }
        
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

        with predictions_right_column:
            st.text("XGBoost:")

            # Creating and showing predicted values and probabilities
            predicted_probability_anomalous = model2.predict_proba(X_df)[:,1]
            predicted_probability_normal = model2.predict_proba(X_df)[:,0]
            predicted_results = model2.predict(X_df)

            predicted_results = map(lambda x: "Normal" if x == 0 else "Anomaly", predicted_results)
            predicted_probability_normal = map(lambda x: f"{round(x*100,1)}%", predicted_probability_normal)
            predicted_probability_anomalous = map(lambda x: f"{round(x*100,1)}%", predicted_probability_anomalous)

            results = {
                "Predicted Behaviour": predicted_results,
                "Probability of being normal": predicted_probability_normal,
                "Probability of being anomalous": predicted_probability_anomalous
            }
        
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

# Showing evaluations for SHAP
def evaluations_shap(model1, model2):
    if model2 is None:

        # Setting up SHAP explainer
        explainer = shap.TreeExplainer(model1)
        shap_values = explainer.shap_values(X_df)

        # Turning decision tree into unicolor graph (not splitting into classes)
        if model1 is dtree:
            shap_values = shap_values[0] * 2

        with centre_column:
            st.text("")
            st.subheader("SHAP Tree Explainer Bar Chart")

            # Plotting
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_df, plot_type='bar', show=False)
            ax.tick_params(axis='x',colors="white")
            ax.tick_params(axis='y',colors="white")
            ax.xaxis.label.set_color("white")
            ax.spines['bottom'].set_color("white")
            ax.spines['left'].set_color("white")
            ax.set_facecolor('#0e1117')
            fig.set_facecolor('#0e1117')
            ax.grid(color='white', alpha = 0.4)
            legend = ax.legend(fontsize=15, loc = "lower right")
            st.pyplot(fig)

    else:
        st.text("")
        st.subheader("SHAP Tree Explainer Bar Chart")

        # Setting up SHAP explainer
        explainer_model1 = shap.TreeExplainer(model1)
        explainer_model2 = shap.TreeExplainer(model2)
        shap_values_model1 = explainer_model1.shap_values(X_df)
        shap_values_model2 = explainer_model2.shap_values(X_df)

        # Turning decision tree into unicolor graph (not splitting into classes)
        shap_values_model1 = shap_values_model1[0] * 2

        # Creating two columns
        shap_left_column, shap_right_column = st.columns([1,1])

        # Plotting
        with shap_left_column:
            st.markdown("**Decision Tree:**")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values_model1, X_df, plot_type='bar', show=False)
            ax.tick_params(axis='x',colors="white")
            ax.tick_params(axis='y',colors="white")
            ax.xaxis.label.set_color("white")
            ax.spines['bottom'].set_color("white")
            ax.spines['left'].set_color("white")
            ax.set_facecolor('#0e1117')
            fig.set_facecolor('#0e1117')
            ax.grid(color='white', alpha = 0.4)
            legend = ax.legend(fontsize=15, loc = "lower right")
            st.pyplot(fig)

        with shap_right_column:
            st.markdown("**XGBoost:**")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values_model2, X_df, plot_type='bar', show=False)
            ax.tick_params(axis='x',colors="white")
            ax.tick_params(axis='y',colors="white")
            ax.xaxis.label.set_color("white")
            ax.spines['bottom'].set_color("white")
            ax.spines['left'].set_color("white")
            ax.set_facecolor('#0e1117')
            fig.set_facecolor('#0e1117')
            ax.grid(color='white', alpha = 0.4)
            legend = ax.legend(fontsize=15, loc = "lower right")
            st.pyplot(fig)

if selected_shap: 
    if selected_dtree and selected_xgboost is False:
        evaluations_shap(dtree, None)

    elif selected_dtree is False and selected_xgboost:
        evaluations_shap(xgboost, None)

    elif selected_xgboost and selected_dtree:
        evaluations_shap(dtree, xgboost)

# Showing evaluations for accuracy and F1 score
def evaluate_score(model1, model2):
    if model2 is None:
        with centre_column:
            st.text("")
            st.subheader("Predictions and Score")
            predicted_probability_anomalous = model1.predict_proba(X_df)[:,1]
            predicted_probability_normal = model1.predict_proba(X_df)[:,0]
            predicted_results = model1.predict(X_df)

            # Finding and printing score
            score = accuracy_score(Y_df, predicted_results)
            st.write(f"Accuracy score: {round(score, 5)}")

            # Finding and printing F1 score
            f1 = f1_score(Y_df, predicted_results)
            st.write(f"F1 Score: {round(f1, 5)}")

            # Creating and showing predicted values and probabilities
            true_results = map(lambda x: "Normal" if x == 0 else "Anomaly", Y_df["Unusual"])
            predicted_results = map(lambda x: "Normal" if x == 0 else "Anomaly", predicted_results)
            predicted_probability_normal = map(lambda x: f"{round(x*100,1)}%", predicted_probability_normal)
            predicted_probability_anomalous = map(lambda x: f"{round(x*100,1)}%", predicted_probability_anomalous)

            results = {
                "True Behaviour": true_results,
                "Predicted Behaviour": predicted_results,
                "Probability of being normal": predicted_probability_normal,
                "Probability of being anomalous": predicted_probability_anomalous
            }

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

    else:
        st.text("")
        st.subheader("Predictions and Score")

        # Creating two columns
        score_left_column, score_right_column = st.columns([1,1])

        with score_left_column:
            st.markdown("**Decision Tree:**")
            predicted_probability_anomalous = model1.predict_proba(X_df)[:,1]
            predicted_probability_normal = model1.predict_proba(X_df)[:,0]
            predicted_results = model1.predict(X_df)

            # Finding and printing score
            score = accuracy_score(Y_df, predicted_results)
            st.write(f"Accuracy score: {round(score, 5)}")

            # Finding and printing F1 score
            f1 = f1_score(Y_df, predicted_results)
            st.write(f"F1 Score: {round(f1, 5)}")

            # Creating and showing predicted values and probabilities
            true_results = map(lambda x: "Normal" if x == 0 else "Anomaly", Y_df["Unusual"])
            predicted_results = map(lambda x: "Normal" if x == 0 else "Anomaly", predicted_results)
            predicted_probability_normal = map(lambda x: f"{round(x*100,1)}%", predicted_probability_normal)
            predicted_probability_anomalous = map(lambda x: f"{round(x*100,1)}%", predicted_probability_anomalous)

            results = {
                "True Behaviour": true_results,
                "Predicted Behaviour": predicted_results,
                "Probability of being normal": predicted_probability_normal,
                "Probability of being anomalous": predicted_probability_anomalous
            }

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

        with score_right_column:
            st.markdown("**XGBoost:**")
            predicted_probability_anomalous = model2.predict_proba(X_df)[:,1]
            predicted_probability_normal = model2.predict_proba(X_df)[:,0]
            predicted_results = model2.predict(X_df)

            # Finding and printing score
            score = accuracy_score(Y_df, predicted_results)
            st.write(f"Accuracy score: {round(score, 5)}")

            # Finding and printing F1 score
            f1 = f1_score(Y_df, predicted_results)
            st.write(f"F1 Score: {round(f1, 5)}")

            # Creating and showing predicted values and probabilities
            true_results = map(lambda x: "Normal" if x == 0 else "Anomaly", Y_df["Unusual"])
            predicted_results = map(lambda x: "Normal" if x == 0 else "Anomaly", predicted_results)
            predicted_probability_normal = map(lambda x: f"{round(x*100,1)}%", predicted_probability_normal)
            predicted_probability_anomalous = map(lambda x: f"{round(x*100,1)}%", predicted_probability_anomalous)

            results = {
                "True Behaviour": true_results,
                "Predicted Behaviour": predicted_results,
                "Probability of being normal": predicted_probability_normal,
                "Probability of being anomalous": predicted_probability_anomalous
            }

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

if selected_score:
    if selected_dtree and selected_xgboost is False:
        evaluate_score(dtree, None)

    elif selected_dtree is False and selected_xgboost:
        evaluate_score(xgboost, None)

    elif selected_dtree and selected_xgboost:
        evaluate_score(dtree, xgboost)

# Showing evaluations for ROC
def evaluate_roc(model1, model2):
    if model2 is None:
        with centre_column:
            st.text("")
            st.subheader("ROC (Receiver Operating Characteristic) Curve")
            predicted_probability = model1.predict_proba(X_df)[:,1]

            false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(Y_df, predicted_probability)
            roc_auc = auc(false_positive_rate, true_positive_rate)

            # Plotting ROC Curve
            fig, ax = plt.subplots()
            ax.plot(false_positive_rate, true_positive_rate, label=f"ROC Curve | AUC = {round(roc_auc,5)}")
            ax.plot([0,1], [0,1], linestyle='--', label="Baseline")
            ax.set_xlabel("False Positive Rate", color='white')
            ax.set_ylabel("True Positive Rate", color='white')
            ax.grid(color='white', alpha=0.4)
            ax.tick_params(axis='x',colors="white")
            ax.tick_params(axis='y',colors="white")
            ax.spines['bottom'].set_color("white")
            ax.spines['left'].set_color("white")
            ax.set_facecolor('#0e1117')
            fig.set_facecolor('#0e1117')
            legend = ax.legend(fontsize=10, loc="lower right")
            st.pyplot(fig)
    else:
        # Creating columns
        roc_left_column, roc_centre_column, roc_right_column = st.columns([1,6,1])

        with roc_centre_column:
            st.text("")
            st.subheader("ROC (Receiver Operating Characteristic) Curve")

            # Plotting ROC Curve with two lines
            predicted_probability_dtree = model1.predict_proba(X_df)[:,1]
            predicted_probability_xgboost = model2.predict_proba(X_df)[:,1]

            false_positive_rate_dtree, true_positive_rate_dtree, roc_thresholds_dtree = roc_curve(Y_df, predicted_probability_dtree)
            false_positive_rate_xgboost, true_positive_rate_xgboost, roc_thresholds_xgboost = roc_curve(Y_df, predicted_probability_xgboost)
            roc_auc_dtree = auc(false_positive_rate_dtree, true_positive_rate_dtree)
            roc_auc_xgboost = auc(false_positive_rate_xgboost, true_positive_rate_xgboost)

            fig, ax = plt.subplots()
            ax.plot(false_positive_rate_dtree, true_positive_rate_dtree, label=f"Decision Tree | AUC = {round(roc_auc_dtree,5)}", color='C1', alpha=1)
            ax.plot(false_positive_rate_xgboost, true_positive_rate_xgboost, label=f"XGBoost | AUC = {round(roc_auc_xgboost,5)}", color='C2', alpha=1,)
            ax.plot([0,1], [0,1], linestyle='--', label="Baseline", color='C0')
            ax.set_xlabel("False Positive Rate", color='white')
            ax.set_ylabel("True Positive Rate", color='white')
            ax.grid(color='white', alpha=0.4)
            ax.tick_params(axis='x',colors="white")
            ax.tick_params(axis='y',colors="white")
            ax.spines['bottom'].set_color("white")
            ax.spines['left'].set_color("white")
            ax.set_facecolor('#0e1117')
            fig.set_facecolor('#0e1117')
            legend = ax.legend(fontsize=10, loc="lower right")
            st.pyplot(fig)


if selected_roc:
    if selected_dtree and selected_xgboost is False:
        evaluate_roc(dtree, None)

    elif selected_dtree is False and selected_xgboost:
        evaluate_roc(xgboost, None)

    elif selected_dtree and selected_xgboost:   
        evaluate_roc(dtree, xgboost)
       
# Showing evaluations for PRD
def evaluate_prd(model1, model2):
    if model2 is None:
        with centre_column:
            st.text("")
            st.subheader("PRD (Precision Recall Display) Curve")
            predicted_probability = model1.predict_proba(X_df)[:,1]

            # Getting PRD and AUC
            precision, recall, prd_threshold = precision_recall_curve(Y_df, predicted_probability)
            prd_auc = average_precision_score(Y_df, predicted_probability)

            # Plotting Precision Recall Display
            fig, ax = plt.subplots()
            ax.plot(recall, precision, label = f"PRD Curve | AUC = {round(prd_auc,5)}")
            ax.set_xlabel("Recall", color = "white")
            ax.set_ylabel("Precision", color = "white")
            ax.tick_params(axis='x',colors="white")
            ax.tick_params(axis='y',colors="white")
            ax.spines['bottom'].set_color("white")
            ax.spines['left'].set_color("white")
            ax.set_facecolor('#0e1117')
            fig.set_facecolor('#0e1117')
            ax.set_ylim(-0.1,1.1)
            ax.grid(color='white', alpha = 0.4)
            legend = ax.legend(fontsize=10, loc = "lower right")
            st.pyplot(fig)
    else:
        # Creating columns
        prd_left_column, prd_centre_column, prd_right_column = st.columns([1,6,1])

        with prd_centre_column:
            st.text("")
            st.subheader("PRD (Precision Recall Display) Curve")
            predicted_probability_model1 = model1.predict_proba(X_df)[:,1]
            predicted_probability_model2 = model2.predict_proba(X_df)[:,1]

            # Getting PRD and AUC
            precision_dtree, recall_dtree, prd_threshold_dtree = precision_recall_curve(Y_df, predicted_probability_model1)
            precision_xgboost, recall_xgboost, prd_threshold_xgboost = precision_recall_curve(Y_df, predicted_probability_model2)
            prd_auc_dtree = average_precision_score(Y_df, predicted_probability_model1)
            prd_auc_xgboost = average_precision_score(Y_df, predicted_probability_model2)

            # Plotting Precision Recall Display
            fig, ax = plt.subplots()
            ax.plot(recall_dtree, precision_dtree, label=f"Decision Tree | AUC = {round(prd_auc_dtree, 5)}", color="C1")
            ax.plot(recall_xgboost, precision_xgboost, label=f"XGBoost | AUC = {round(prd_auc_xgboost, 5)}", color="C2")
            ax.set_xlabel("Recall", color="white")
            ax.set_ylabel("Precision", color="white")
            ax.tick_params(axis='x',colors="white")
            ax.tick_params(axis='y',colors="white")
            ax.spines['bottom'].set_color("white")
            ax.spines['left'].set_color("white")
            ax.set_facecolor('#0e1117')
            fig.set_facecolor('#0e1117')
            ax.set_ylim(-0.1,1.1)
            ax.grid(color='white', alpha=0.4)
            legend = ax.legend(fontsize=10, loc="lower right")
            st.pyplot(fig)


if selected_prd:
    if selected_dtree and selected_xgboost is False:
        evaluate_prd(dtree, None)

    if selected_dtree is False and selected_xgboost:
        evaluate_prd(xgboost, None)

    if selected_dtree and selected_xgboost:
        evaluate_prd(dtree, xgboost)
        