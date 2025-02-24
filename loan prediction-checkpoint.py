# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Import the dataset
loan_data = pd.read_csv('"F:\loan-predictionUC.csv.xlsx"') 

# Step 3: Understand the data
print(loan_data.head())  
print(loan_data.info())  
print(loan_data.describe()) 

# Step 4: Deal with missing values if any


# Example:
loan_data['LoanAmount'].fillna(loan_data['LoanAmount'].mean(), inplace=True)
loan_data['Loan_Amount_Term'].fillna(loan_data['Loan_Amount_Term'].mode()[0], inplace=True)
loan_data['Credit_history'].fillna(0, inplace=True)  # Assuming 0 means no credit history

# Step 5: Data Visualization (Optional)
# You can visualize the data to gain insights, e.g., plot histograms, count plots, etc.

# Example:
plt.figure(figsize=(10, 6))
loan_data['Loan_status'].value_counts().plot(kind='bar')
plt.title('Loan Status Distribution')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()

# Step 6: Divide the dataset into training and test datasets
X = loan_data.drop(columns=['Loan_id', 'Loan_status'])  # Features
y = loan_data['Loan_status']  # Target variable

# Use label encoding to convert categorical features to numerical
label_encoder = LabelEncoder()
X['Gender'] = label_encoder.fit_transform(X['Gender'])
X['Married'] = label_encoder.fit_transform(X['Married'])
X['Education'] = label_encoder.fit_transform(X['Education'])
X['Self-employed'] = label_encoder.fit_transform(X['Self-employed'])
X['Property_area'] = label_encoder.fit_transform(X['Property_area'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Build the machine learning model (Random Forest Classifier in this example)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 8: Fit the model on the training dataset
model.fit(X_train, y_train)

# Step 9: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# You can further evaluate the model using metrics like confusion matrix, classification report, etc.
print(classification_report(y_test, y_pred))
