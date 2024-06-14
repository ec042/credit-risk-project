import pytest
import pickle
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

dataset_url = "https://drive.google.com/uc?export=download&id=1u2fbeAC4wgT1uu4_gnrBABU8v6GcV1bw"

@pytest.fixture(scope="module")
def model_and_encoders():
    # Read the CSV file from the direct download URL
    CreditRiskData = pd.read_csv(dataset_url)
    print('Shape before deleting duplicate values:', CreditRiskData.shape)
    assert not CreditRiskData.empty, "Dataset should not be empty"
    
    # Removing duplicate rows if any
    CreditRiskData = CreditRiskData.dropna()
    print('Shape After deleting duplicate values:', CreditRiskData.shape)
    
    # Print sample data
    print(CreditRiskData.head())
    print(CreditRiskData.info())
    print(CreditRiskData.describe())
    
    # Finding how many missing values are there for each column
    print(CreditRiskData.isnull().sum())
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['Name', 'Occupation', 'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score']
    for column in categorical_columns:
        le = LabelEncoder()
        CreditRiskData[column] = le.fit_transform(CreditRiskData[column])
        label_encoders[column] = le

    # Scale numerical variables
    scaler = StandardScaler()
    numerical_columns = ['Age', 'SSN', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
                         'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
                         'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
                         'Credit_History_Age', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
    CreditRiskData[numerical_columns] = scaler.fit_transform(CreditRiskData[numerical_columns])
    
    # Split data into train and test sets
    X = CreditRiskData.drop(columns=['ID', 'Customer_ID', 'Credit_Score'])
    y = CreditRiskData['Credit_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the logistic regression model
    model = LogisticRegression(max_iter=5000, solver='lbfgs', multi_class='multinomial')
    model.fit(X_train, y_train)

    return model, label_encoders, scaler, categorical_columns, numerical_columns

def test_read_csv(model_and_encoders):
    model, label_encoders, scaler, categorical_columns, numerical_columns = model_and_encoders
    
    # Assert that the model has been trained
    assert model is not None, "Model training failed"

    # Evaluate the model on the test data within this test function if needed

def test_load_test_data_and_predict(model_and_encoders):
    model, label_encoders, scaler, categorical_columns, numerical_columns = model_and_encoders

    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'test.csv')
    test_data = pd.read_csv(csv_path)

    for column in categorical_columns[:-1]:  # Exclude 'Credit_Score'
        le = label_encoders[column]
        test_data[column] = le.transform(test_data[column])

    test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])

    X_test_final = test_data.drop(columns=['ID', 'Customer_ID'])
    y_test_pred = model.predict(X_test_final)
    y_test_pred_proba = model.predict_proba(X_test_final)

    y_test_pred_decoded = label_encoders['Credit_Score'].inverse_transform(y_test_pred)
    results = test_data[['ID', 'Customer_ID']].copy()
    results['Predicted_Credit_Score'] = y_test_pred_decoded
    results.to_csv("predicted_credit_scores.csv", index=False)
    print("Predictions saved to predicted_credit_scores.csv")



