# Anomaly detection in LTE activity

## Objective
1. Apply 3 supervised learning models to detect anormalies in LTE activity dataset.
    * Decision Tree
    * Random Forest
    * XGBoost
2. Develop a simple GUI/web application to allow user interactivity with trained models. 

## Deliverables
Perform the following seperately:
1. Exploratory Data Analysis in Jupyter Notebook
2. End-to-end Machine Learning Pipeline (MLP) in Python Scripts
3. Simple GUI/web application in Python:
    - input: file containing test data
    - output: show model evaluation results  

Reproducibility for MLP is important:
* Don't develop MLP in an interactive notebook.
* Pipeline should be configurable to enable easy experimentation of different. You can consider the usage of a config file.  

<p>
  <img src="MLP.JPG" alt="pipe_flow">
</p>

In final report & presentation: Provide descriptions:
1. Appropriate data preprocessing and feature engineering
2. Appropriate use and optimization of algorithms/models
3. Explanation for the choice of algorithms/models
4. Explanation for the choice of evaluation metrics
    - Compare the 3 trained models using evaluation metrics  

## Suggested Reading
Book: Machine Learning Bookcamp  
Author: Alexey Grigorev  

| Chapter | Time Period |
|----------|----------|
| Appendix C, Appendix D, 1, 2 (EDA sections) | 11 Sep - 22 Sep |
| 3 (EDA sections) | 22 Sep - 6 Oct |
| 4 | 9 Oct - 13 Oct |
| 6 | 16 Oct - 17 Nov |
| 5 (Pickle, Flask/FastAPI + Streamlit) | 20 Nov - 8 Dec |  


Book: Effective XGBoost  
Author: Matt Harrison  

| Chapter | Time Period |
|----------|----------|
| 4 - Tree Creation | 18 Sep - 22 Sep |
| 5 - Stumps| 25 Sep - 29 Sep |
| 6 - Model Complexity & Hyperparameters| 2 Oct - 6 Oct |
| 7 - Tree Hyperparameters| 9 Oct - 13 Oct |
| 8 - Random Forest | 16 Oct - 20 Oct |
| 9 - XGBoost | 23 Oct -27 Oct |
| 10 - Early Stopping | 30 Oct - 3 Nov |
| 11 - XGBoost Hyperparameters| 6 Nov - 10 Nov |
| 12 - Hyperopt | 13 Nov - 17 Nov |  
| 13 - Step-wise tuning with Hyperopt | 20 Nov - 24 Nov |  

Data source: https://www.kaggle.com/competitions/anomaly-detection-in-4g-cellular-networks/overview