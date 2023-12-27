# Updates
# Info: change description of info, add referencing to flaticon
# Use st.divider where appropriate
# Add a part in info or sidebar (sidebar can be like, this app acts as an interface for users to access the two trained ML models, it processes raw data, ....) about the actual MLP, talk about how the website does all the backend feature engineering and stuff

# Importing libraries
import streamlit as st
import numpy as np 
import pandas as pd 
import pickle
import matplotlib.pyplot as plt
import shap
from streamlit_shap import st_shap
from streamlit_pills import pills
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix, classification_report, precision_recall_curve, PrecisionRecallDisplay, average_precision_score

# Importing models
with open('models/train_dtree_ft2.pkl', 'rb') as model_file:
    dtree = pickle.load(model_file)
with open('models/train_xgboost_ft2.pkl', 'rb') as model_file:
    xgboost = pickle.load(model_file)

# Streamlit configuration
st.set_page_config(
    layout="centered",
    page_title="Anomaly Detection in LTE Network",
    page_icon="📡",
    initial_sidebar_state="collapsed"
)

# Creating Sidebar
sidebar_left_column, sidebar_right_column = st.sidebar.columns([1,2.5]) # Create columns for title
with sidebar_left_column:
    st.image("images/exploration.png")
with sidebar_right_column:
    st.header("Anomaly Detection in LTE Network")
st.sidebar.divider()
st.sidebar.subheader("About")
st.sidebar.markdown("This app acts as an interface to access two machine learning models created to predict radio cell behaviour. It also facilitates the evaluation of these models. ")
st.sidebar.subheader("References")
st.sidebar.markdown("Images used from [Flaticon](https://www.flaticon.com/free-icons/detection). Emojis and icons used from [Emojipedia](https://emojipedia.org/).")
st.sidebar.divider()
st.sidebar.markdown("This app was made for Research@YDSP 2023, the project poster and report can be accessed at the [official DSTA website.](https://www.dsta.gov.sg/ydsp/projects/)")
st.sidebar.divider()

# Creating columns to centralize title
title_first_column, title_second_column, title_third_column, title_fourth_column = st.columns([0.2, 1, 0.01, 2.8])

# Creating main title, brief description and navigation bar
with title_second_column:
    st.image("images/detection1.png", width=130)

with title_fourth_column:
    st.title("Anomaly Detection in LTE Network")

main_tab, telemetry_data_tab, info_tab = st.tabs(["Main", "Telemetry Data", "Info"])

# "Main" tab 
with main_tab:
    predictor_file_error = False
    outcome_file_error = False
    st.write("")
    st.subheader("1️⃣ Input Telemetry Data")
    uploaded_X_data = st.file_uploader("Input predictor features")
    uploaded_Y_data = st.file_uploader("Input outcome feature / result")

    # Creates and processes dataframes if files are input
    if uploaded_X_data is not None:
        X_df_raw = pd.read_csv(uploaded_X_data)

        # Feature engineering preprocessor
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

        # Pipeline for log transformation and standard scaler
        Pipe = Pipeline(steps = [
            ('log', Log_Transformer),
            ('scale', StandardScaler())
        ])

        # Creating Dataframes
        X_train = pd.read_csv("data/X_train.csv")
        Y_train = pd.read_csv("data/Y_train.csv")

        # Turning time into a datetime type
        X_train['Time'] = pd.to_datetime(X_train['Time'], format = '%H:%M')

        # Creating new columns for meanUE_UL_encoded and meanUE_DL_encoded (encoding will be done later)
        X_train['meanUE_UL_encoded'] = X_train['meanUE_UL']
        X_train['meanUE_DL_encoded'] = X_train['meanUE_DL']

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
        
        try:
            # Adding Index
            X_df_raw.insert(0, 'Unnamed: 0', np.arange(0, X_df_raw.shape[0], 1))

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
                'Cell ID: 8BLTE', 'Cell ID: 5CLTE', 'Cell ID: 6BLTE', 'Time', 'Mean UE devices encoded (uplink)', 
                'Mean UE devices encoded (downlink)', 'Percentage of PRB usage (uplink)', 'Percentage of PRB usage (downlink)',
                'Mean carried traffic (downlink)', 'Mean carried traffic (uplink)', 'Max carried traffic (downlink)', 
                'Max carried traffic (uplink)', 'Mean UE devices (downlink)', 'Mean UE devices (uplink)', 
                'Max UE devices (downlink)', 'Max UE devices (uplink)', 'Max UE devices (uplink and downlink)', 
                'Outdated_Index']

            X_df.columns = Columns

            # Dropping index
            X_df.drop(X_df.columns[[-1]], axis = 1, inplace = True)

            # Removing maxUE_UL, maxUE_UL+DL, Time (insignificant features)
            X_df = X_df.drop(["Max UE devices (uplink)", "Max UE devices (uplink and downlink)", "Time"], axis=1)

            predictor_file_error = False

        except:
            st.error("Error: Predictor features file incompatible, did you mean to input the outcome feature file?")
            predictor_file_error = True

    else:
        pass

    if uploaded_Y_data is not None:
        Y_df = pd.read_csv(uploaded_Y_data)
        if Y_df.shape[1] > 1:
            st.error("Error: Outcome feature file incompatible, did you mean to input the predictor feature file?")
            outcome_file_error = True
        else:
            outcome_file_error = False
    else:
        pass

    # Creating second part of "Main" tab (Model Results)
    selected_evaluation = ""

    if uploaded_X_data is not None and uploaded_Y_data is not None and predictor_file_error is False and outcome_file_error is False:
        if X_df.shape[0] != Y_df.shape[0]: # Error message to check that files have same length
            st.error("Error: Files do not have same length")
        else:
            st.subheader('2️⃣ Model Results')

            with st.container(border=True):
                st.markdown("**Selected Machine Learning Model:**")

                # Creating columns for checkboxes
                checkbox_left_column, checkbox_right_column = st.columns([1,1])

                with checkbox_left_column:
                    selected_dtree = st.checkbox("Decision Tree")
                with checkbox_right_column:
                    selected_xgboost = st.checkbox("XGBoost")

                if selected_dtree or selected_xgboost:
                    st.write("")
                    st.markdown("**Evaluation Metrics:**")
                    selected_evaluation = pills("", ["Predictions", "SHAP", "Accuracy and F1 Score", "ROC", "PRD"], label_visibility="collapsed")

                else:
                    pass

            st.divider()

    elif uploaded_X_data is not None and uploaded_Y_data is None and predictor_file_error is False and outcome_file_error is False: 
        st.markdown(" ")
        st.subheader('2️⃣ Model Results')

        with st.container(border=True):
            st.markdown("**Selected Machine Learning Model:**")

            # Creating columns for checkboxes
            checkbox_left_column, checkbox_right_column = st.columns([1,1])

            with checkbox_left_column:
                selected_dtree = st.checkbox("Decision Tree")
            with checkbox_right_column:
                selected_xgboost = st.checkbox("XGBoost")

            if selected_dtree or selected_xgboost:
                st.markdown("**Evaluation Metrics:**")
                selected_evaluation = pills("", ["Predictions", "SHAP"], label_visibility="collapsed")
                st.warning('🔒 Upload outcome feature / results as well for full model evaluation')

            else:
                pass
            
        st.divider()

    elif predictor_file_error is False and outcome_file_error is False:
        st.info('☝️ Upload CSV files')

    else:
        pass

    # Showing evaluations for Predictions 
    def evaluations_predictions(model1, model2):
        if model2 is None:
            st.subheader("Model Predictions")

            # Creating and showing predicted values and probabilities
            predicted_probability_anomalous = model1.predict_proba(X_df)[:,1]
            predicted_results = model1.predict(X_df)

            predicted_results = map(lambda x: "Normal" if x == 0 else "Anomaly", predicted_results)
            predicted_probability = map(lambda x: f"{round(x*100,1)}%" if x > 0.5 else f"{round((1-x)*100)}%", predicted_probability_anomalous)
                
            if uploaded_Y_data is not None:
                actual_results = map(lambda x: "Normal" if x == 0 else "Anomalous", Y_df["Unusual"])
                results = {
                    "Predicted Behaviour": predicted_results,
                    "Actual Behaviour": actual_results,
                    "Predicted Probability": predicted_probability
                }
               
            else:
                results = {
                    "Predicted Behaviour": predicted_results,
                    "Predicted Probability": predicted_probability
                }

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            st.divider()

        else: 
            st.subheader("Model Predictions")

            # Creating two columns
            predictions_left_column, predictions_right_column = st.columns([1,1])

            with predictions_left_column:
                st.markdown("**Decision Tree:**")

                # Creating and showing predicted values and probabilities
                predicted_probability_anomalous = model1.predict_proba(X_df)[:,1]
                predicted_results = model1.predict(X_df)

                predicted_results = map(lambda x: "Normal" if x == 0 else "Anomaly", predicted_results)
                predicted_probability = map(lambda x: f"{round(x*100,1)}%" if x > 0.5 else f"{round((1-x)*100)}%", predicted_probability_anomalous)
                
                if uploaded_Y_data is not None:
                    actual_results = map(lambda x: "Normal" if x == 0 else "Anomalous", Y_df["Unusual"])
                    results = {
                        "Predicted Behaviour": predicted_results,
                        "Actual Behaviour": actual_results,
                        "Predicted Probability": predicted_probability
                    }
                
                else:
                    results = {
                        "Predicted Behaviour": predicted_results,
                        "Predicted Probability": predicted_probability
                    }
            
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)

            with predictions_right_column:
                st.markdown("**XGBoost:**")

                # Creating and showing predicted values and probabilities
                predicted_probability_anomalous = model1.predict_proba(X_df)[:,1]
                predicted_results = model1.predict(X_df)

                predicted_results = map(lambda x: "Normal" if x == 0 else "Anomaly", predicted_results)
                predicted_probability = map(lambda x: f"{round(x*100,1)}%" if x > 0.5 else f"{round((1-x)*100)}%", predicted_probability_anomalous)
                
                if uploaded_Y_data is not None:
                    actual_results = map(lambda x: "Normal" if x == 0 else "Anomalous", Y_df["Unusual"])
                    results = {
                        "Predicted Behaviour": predicted_results,
                        "Actual Behaviour": actual_results,
                        "Predicted Probability": predicted_probability
                    }
                
                else:
                    results = {
                        "Predicted Behaviour": predicted_results,
                        "Predicted Probability": predicted_probability
                    }
            
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
            st.divider()

    if selected_evaluation == "Predictions": 
        if selected_dtree and selected_xgboost is False:
            evaluations_predictions(dtree, None)

        elif selected_dtree is False and selected_xgboost:
            evaluations_predictions(xgboost, None)

        elif selected_xgboost and selected_dtree:
            evaluations_predictions(dtree, xgboost)

        else:
            pass
    else:
        pass

    # Showing evaluations for SHAP
    def evaluations_shap(model1, model2):
        if model2 is None:

            # Setting up SHAP explainer
            explainer = shap.TreeExplainer(model1)
            shap_values = explainer.shap_values(X_df)

            # Turning decision tree into unicolor graph (not splitting into classes)
            if model1 is dtree:
                shap_values = shap_values[0] * 2
                color = "C1"
            else:
                color = "C2"

            st.subheader("SHAP Tree Explainer Bar Chart")

            # Plotting
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_df, plot_type='bar', show=False, color=color)
            ax.tick_params(axis='x',colors="white")
            ax.tick_params(axis='y',colors="white")
            ax.xaxis.label.set_color("white")
            ax.spines['bottom'].set_color("white")
            ax.spines['left'].set_color("white")
            ax.set_facecolor('#0e1117')
            fig.set_facecolor('#0e1117')
            ax.set_xlabel("Mean |SHAP Value|")
            ax.grid(color='white', alpha = 0.4)
            legend = ax.legend(fontsize=15, loc = "lower right")
            st.pyplot(fig)
            st.divider()

        else:
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
                shap.summary_plot(shap_values_model1, X_df, plot_type='bar', show=False, color="C1")
                ax.tick_params(axis='x',colors="white")
                ax.tick_params(axis='y',colors="white")
                ax.xaxis.label.set_color("white")
                ax.spines['bottom'].set_color("white")
                ax.spines['left'].set_color("white")
                ax.set_facecolor('#0e1117')
                fig.set_facecolor('#0e1117')
                ax.set_xlabel("Mean |SHAP Value|")
                ax.grid(color='white', alpha = 0.4)
                legend = ax.legend(fontsize=15, loc = "lower right")
                st.pyplot(fig)

            with shap_right_column:
                st.markdown("**XGBoost:**")
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values_model2, X_df, plot_type='bar', show=False, color="C2")
                ax.tick_params(axis='x',colors="white")
                ax.tick_params(axis='y',colors="white")
                ax.xaxis.label.set_color("white")
                ax.spines['bottom'].set_color("white")
                ax.spines['left'].set_color("white")
                ax.set_facecolor('#0e1117')
                fig.set_facecolor('#0e1117')
                ax.set_xlabel("Mean |SHAP Value|")
                ax.grid(color='white', alpha = 0.4)
                legend = ax.legend(fontsize=15, loc = "lower right")
                st.pyplot(fig)
            st.divider()

    if selected_evaluation == "SHAP": 
        if selected_dtree and selected_xgboost is False:
            evaluations_shap(dtree, None)

        elif selected_dtree is False and selected_xgboost:
            evaluations_shap(xgboost, None)

        elif selected_xgboost and selected_dtree:
            evaluations_shap(dtree, xgboost)
        else:
            pass
    else:
        pass

    # Showing evaluations for accuracy and F1 score
    def evaluate_score(model1, model2):
        if model2 is None:
            st.subheader("Accuracy and F1 Score")
            predicted_probability_anomalous = model1.predict_proba(X_df)[:,1]
            predicted_probability_normal = model1.predict_proba(X_df)[:,0]
            predicted_results = model1.predict(X_df)

            # Creating two columns
            score_left_column, score_right_column = st.columns([1,1])

            # Finding and printing score
            score = accuracy_score(Y_df, predicted_results)
            with score_left_column:
                st.metric("Accuracy", round(score, 5))

            # Finding and printing F1 score
            f1 = f1_score(Y_df, predicted_results)
            with score_right_column:
                st.metric("F1 Score", round(f1, 5))

            st.divider()

        else:
            st.subheader("Accuracy and F1 Score")

            # Creating two columns
            score_title_left_column, score_title_right_column = st.columns([1,1])

            # Creating four columns
            score_first_column, score_second_column, score_third_column, score_fourth_column = st.columns([1,1,1,1])

            with score_title_left_column:
                st.markdown("**Decision Tree:**")
                
            predicted_probability_anomalous = model1.predict_proba(X_df)[:,1]
            predicted_probability_normal = model1.predict_proba(X_df)[:,0]
            predicted_results = model1.predict(X_df)

            # Finding and printing score
            model1_score = accuracy_score(Y_df, predicted_results)
            with score_first_column:
                st.metric("Accuracy", round(model1_score, 5))

            # Finding and printing F1 score
            model1_f1 = f1_score(Y_df, predicted_results)
            with score_second_column:
                st.metric("F1 Score", round(model1_f1, 5))

            with score_title_right_column:
                st.markdown("**XGBoost:**")

            predicted_probability_anomalous = model2.predict_proba(X_df)[:,1]
            predicted_probability_normal = model2.predict_proba(X_df)[:,0]
            predicted_results = model2.predict(X_df)

            # Finding and printing score
            model2_score = accuracy_score(Y_df, predicted_results)
            with score_third_column:
                st.metric("Accuracy", round(model2_score, 5), delta=round(model2_score-model1_score,5))

            # Finding and printing F1 score
            model2_f1 = f1_score(Y_df, predicted_results)
            with score_fourth_column:
                st.metric("F1 Score", round(model2_f1, 5), delta=round(model2_f1-model1_f1,5))

            st.divider()

    if selected_evaluation == "Accuracy and F1 Score":
        if selected_dtree and selected_xgboost is False:
            evaluate_score(dtree, None)

        elif selected_dtree is False and selected_xgboost:
            evaluate_score(xgboost, None)

        elif selected_dtree and selected_xgboost:
            evaluate_score(dtree, xgboost)

        else:
            pass
    else:
        pass

    # Showing evaluations for ROC
    def evaluate_roc(model1, model2):
        if model2 is None:
            st.subheader("Receiver Operating Characteristic (ROC)")

            if model1 is dtree:
                color = "C1"
            else:
                color = "C2"

            predicted_probability = model1.predict_proba(X_df)[:,1]
            false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(Y_df, predicted_probability)
            roc_auc = auc(false_positive_rate, true_positive_rate)

            # Plotting ROC Curve
            fig, ax = plt.subplots()
            ax.plot(false_positive_rate, true_positive_rate, label=f"ROC Curve | AUC = {round(roc_auc,5)}", color=color)
            ax.plot([0,1], [0,1], linestyle='--', label="Baseline", color="C0")
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
            st.divider()

        else:
            st.subheader("Receiver Operating Characterstic (ROC)")

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
            st.divider()


    if selected_evaluation == "ROC":
        if selected_dtree and selected_xgboost is False:
            evaluate_roc(dtree, None)

        elif selected_dtree is False and selected_xgboost:
            evaluate_roc(xgboost, None)

        elif selected_dtree and selected_xgboost:   
            evaluate_roc(dtree, xgboost)

        else:
            pass
    else:
        pass
        
    # Showing evaluations for PRD
    def evaluate_prd(model1, model2):
        if model2 is None:
            st.subheader("Precision Recall Display (PRD)")

            if model1 is dtree:
                shap_values = shap_values[0] * 2
                color = "C1"

            else:
                color = "C2"

            predicted_probability = model1.predict_proba(X_df)[:,1]

            # Getting PRD and AUC
            precision, recall, prd_threshold = precision_recall_curve(Y_df, predicted_probability)
            prd_auc = average_precision_score(Y_df, predicted_probability)

            # Plotting Precision Recall Display
            fig, ax = plt.subplots()
            ax.plot(recall, precision, label = f"PRD Curve | AUC = {round(prd_auc,5)}", color=color)
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
            st.divider()

        else:
            st.subheader("Precision Recall Display (PRD)")

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
            st.divider()

    if selected_evaluation == "PRD":
        if selected_dtree and selected_xgboost is False:
            evaluate_prd(dtree, None)

        if selected_dtree is False and selected_xgboost:
            evaluate_prd(xgboost, None)

        if selected_dtree and selected_xgboost:
            evaluate_prd(dtree, xgboost)

        else:
            pass
    else:
        pass

# "Sample Data" tab

# Importing test data and converting it into suitable form
df_X_test = pd.read_csv("data/X_test.csv")
df_Y_test = pd.read_csv("data/Y_test.csv")
df_X_test = df_X_test.iloc[:, 1:] # Removing index
df_Y_test = df_Y_test.iloc[:, 1:]

df_X_sample = pd.read_csv("data/X_test_sample.csv")
df_Y_sample = pd.read_csv("data/Y_test_sample.csv")

csv_X_sample = df_X_sample.to_csv(index=False).encode('utf-8')
csv_Y_sample = df_Y_sample.to_csv(index=False).encode('utf-8')

# Main body of "Telemetry Data" tab
with telemetry_data_tab:
    st.subheader("Download Sample")
    st.write("No telemetry data? Readily available samples can be downloaded here as CSV files. The sample comes from the testing dataset which was not used to train the models.")
    button_left_column, button_right_column = st.columns([1,1]) # Creating columns for download buttons
    with button_left_column:
        st.download_button(label="Download predictor features", data=csv_X_sample, file_name="sample_predictor_data.csv", use_container_width=True)
    with button_right_column:
        st.download_button(label="Download outcome features", data=csv_Y_sample, file_name="sample_outcome_data.csv", use_container_width=True)

    st.divider()
    st.subheader("Custom Download")
    
    with st.container(border=True):
        predictor_file_name = st.text_input("File name (predictor features):", value="custom_predictor_data.csv")
        outcome_file_name = st.text_input("File name (outcome feature):", value="custom_outcome_data.csv")
        rows = st.slider("Select rows from the testing dataset:", 1, 9158, (1, 101))
        sample_size = rows[1] - rows[0]
        st.write(f"Sample size: {sample_size} ")

        if sample_size <= 50:
            st.warning("Sample size is too low, model evaluations may be off.")
        elif sample_size >= 3000:
            st.warning("Sample size is too high, certain evaluation metrics will take a long time to process")

    # Creating custom dataframe
    df_X_custom = df_X_test.iloc[rows[0]:rows[1], :]
    df_Y_custom = df_Y_test.iloc[rows[0]:rows[1], :]

    csv_X_custom = df_X_custom.to_csv(index=False).encode('utf-8')
    csv_Y_custom = df_Y_custom.to_csv(index=False).encode('utf-8')

    custom_button_left_column, custom_button_right_column = st.columns([1,1]) # Creating columns for download buttons
    with custom_button_left_column:
        st.download_button(label="Download predictor features", data=csv_X_custom, file_name=predictor_file_name, use_container_width=True)
    with custom_button_right_column:
        st.download_button(label="Download outcome features", data=csv_Y_custom, file_name=outcome_file_name, use_container_width=True)

    st.divider()

# "Info" tab
with info_tab:
    st.subheader("📖 About the Project")
    st.write("Next generation 4G and 5G cellular networks ask for a more efficient and dynamic management of the scarce and expensive radio resources. Network operators must be therefore be capable of anticipating to variations in users’ traffic demands.")
    st.write("As such, the project aims to:")
    st.markdown("1. Explore the possibilities of ML to detect abnormal behaviors in the utilization of the network")
    st.markdown("2. Analyze a [dataset](https://www.kaggle.com/competitions/anomaly-detection-in-4g-cellular-networks/overview) of a past 4G LTE deployment and use it to train two ML models capable of classifying samples of current activity as:")
    st.markdown("* (a) **Normal** activity, corresponding to behaviour of any day at that time, therefore, no re-configuration or redistribution of resources is needed.")
    st.markdown("* (b) **Unusual** activity, which differs from the behavior usually observed at that time of day and should trigger a reconfiguration of the base station.")
    st.markdown("3. Develop an app that acts as an interface for users to utilise the ML models to create predictions and evaluate the performance of the models")

    st.subheader('⚙️ Project Methodology')
    st.image("images/MLP.png", caption="Machine Learning Pipeline")
    st.write("The two supervised classification models being used in this project are:")
    st.markdown("1. [Decision Tree](https://scikit-learn.org/stable/modules/tree.html) model") 
    st.markdown("2. [XGBoost](https://xgboost.readthedocs.io/en/latest/index.html) (eXtreme Gradient Boosting) model")
    st.markdown("The project underwent different phases, including exploratory data analysis, feature engineering, hyperparameter tuning and model evaluation. Further information on the project can be found in the original poster and report submitted for [Research@YDSP 2023](https://www.dsta.gov.sg/ydsp/projects/).")
    st.divider()

# Disables horizontal sliding on mobile
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")