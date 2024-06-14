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

dataset_url = "https://drive.google.com/file/d/1u2fbeAC4wgT1uu4_gnrBABU8v6GcV1bw/view?usp=sharing"

def test_read_csv():
    try:
        CreditRiskData = pd.read_csv(dataset_url)
        print('Shape before deleting duplicate values:', CreditRiskData.shape)
        assert not CreditRiskData.empty, "Dataset should not be empty"
        
        # Proceed with the rest of the code
        # Removing duplicate rows if any
        CreditRiskData = CreditRiskData.dropna()
        print('Shape After deleting duplicate values:', CreditRiskData.shape)

        # Print sample data
        print(CreditRiskData.head())
        print('Shape before deleting duplicate values:', CreditRiskData.shape)
        # Start observing the Quantitative/Categorical/Qualitative variables
        print(CreditRiskData.head())
        print(CreditRiskData.info())
        print(CreditRiskData.describe())

        # Removing duplicate rows if any
        CreditRiskData=CreditRiskData.dropna()
        print('Shape After deleting duplicate values:', CreditRiskData.shape)
        # Printing sample data
        
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
        
        # Evaluate the model on the test data
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        print("ROC-AUC Score:", roc_auc)

        for i in range(len(label_encoders['Credit_Score'].classes_)):
            fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
            plt.plot(fpr, tpr, lw=2, label='class %d (area = %0.2f)' % (i, roc_auc_score(y_test == i, y_pred_proba[:, i])))
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for Multi-Class')
        plt.legend(loc="lower right")
        plt.show()
    except Exception as e:
        pytest.fail(f"Failed to process dataset from {dataset_url}: {e}")

'''# Load test data
current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to 'test.csv'
csv_path = os.path.join(current_dir, '..', 'test.csv')
    # Load the CSV file
test_data = pd.read_csv(csv_path)
# Apply label encoders
for column in categorical_columns[:-1]:  # Exclude 'Credit_Score'
    le = label_encoders[column]
    test_data[column] = le.transform(test_data[column])

# Apply scaler
test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])

# Remove unnecessary columns
X_test_final = test_data.drop(columns=['ID', 'Customer_ID'])

# Make predictions
y_test_pred = model.predict(X_test_final)
y_test_pred_proba = model.predict_proba(X_test_final)

# Decode the predicted labels
y_test_pred_decoded = label_encoders['Credit_Score'].inverse_transform(y_test_pred)

# Prepare the result DataFrame
results = test_data[['ID', 'Customer_ID']].copy()
results['Predicted_Credit_Score'] = y_test_pred_decoded

# Save to CSV
results.to_csv("predicted_credit_scores.csv", index=False)
print("Predictions saved to predicted_credit_scores.csv")'''
def test_load_test_data_and_predict():
    global model
    if model is None:
        pytest.fail("Model is not trained. Please run the test_read_csv test first.")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, '..', 'test.csv')
        test_data = pd.read_csv(csv_path)

        # Apply label encoders
        for column in categorical_columns[:-1]:  # Exclude 'Credit_Score'
            le = label_encoders[column]
            test_data[column] = le.transform(test_data[column])

        # Apply scaler
        test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])

        # Remove unnecessary columns
        X_test_final = test_data.drop(columns=['ID', 'Customer_ID'])

        # Make predictions
        y_test_pred = model.predict(X_test_final)
        y_test_pred_proba = model.predict_proba(X_test_final)

        # Decode the predicted labels
        y_test_pred_decoded = label_encoders['Credit_Score'].inverse_transform(y_test_pred)

        # Prepare the result DataFrame
        results = test_data[['ID', 'Customer_ID']].copy()
        results['Predicted_Credit_Score'] = y_test_pred_decoded

        # Save to CSV
        results.to_csv("predicted_credit_scores.csv", index=False)
        print("Predictions saved to predicted_credit_scores.csv")
    except Exception as e:
        pytest.fail(f"Failed to process test data: {e}")



