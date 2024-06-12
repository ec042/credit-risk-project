import pickle
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
CreditRiskData=  pd.read_csv(r"C:/Users/Dell/New folder/credit/train.csv")
print('Shape before deleting duplicate values:', CreditRiskData.shape)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


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


label_encoders = {}
categorical_columns = ['Name', 'Occupation', 'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score']

for column in categorical_columns:
    le = LabelEncoder()
    CreditRiskData[column] = le.fit_transform(CreditRiskData[column])
    label_encoders[column] = le
scaler = StandardScaler()
numerical_columns = ['Age', 'SSN', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
                     'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
                     'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
                     'Credit_History_Age', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

CreditRiskData[numerical_columns] = scaler.fit_transform(CreditRiskData[numerical_columns])
X = CreditRiskData.drop(columns=['ID', 'Customer_ID', 'Credit_Score'])
y = CreditRiskData['Credit_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train the logistic regression model with increased iterations
model = LogisticRegression(max_iter=5000, solver='lbfgs', multi_class='multinomial')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Ensure the shape of y_pred_proba is correct
print("Shape of y_pred_proba:", y_pred_proba.shape)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC-AUC Score and Curve for multi-class classification
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
print("ROC-AUC Score:", roc_auc)

# Plot ROC Curve for each class
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




