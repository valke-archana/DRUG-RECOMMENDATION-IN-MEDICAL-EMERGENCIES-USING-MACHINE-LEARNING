import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier


from sklearn.preprocessing import LabelEncoder





from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

# Load your dataset
df = pd.read_csv("Training.csv")
df.drop('Unnamed: 133',axis=1,inplace=True)
# Check sparsity (percentage of zero or NaN values in symptom columns)
sparsity = (df.iloc[:, :-1] == 0).sum().sum() / df.iloc[:, :-1].size  # Assuming last column is the target

print(f"Sparsity: {sparsity * 100:.2f}%")

df.iloc[:,:-1]

# Assuming the last column is 'prognosis' or 'disease'
prognosis_col = 'prognosis'  # Change to the actual column name if different
symptom_columns = df.columns.difference([prognosis_col])  # All columns except prognosis

# Group by prognosis and sum symptoms
grouped_df = df.groupby(prognosis_col)[symptom_columns].sum().reset_index()

# Add a total symptoms column
grouped_df['total_symptoms'] = grouped_df.iloc[:, 1:].sum(axis=1)

# Sort by total symptoms (Descending)
grouped_df = grouped_df.sort_values(by='total_symptoms', ascending=False)

# Display result
print(grouped_df[['prognosis','total_symptoms']])

"""# RandomForest classifier -Before feature selection(15 features)"""

# Separate features and target
X = df.drop(columns=["prognosis"])
y = df["prognosis"]

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importance
importances = model.feature_importances_

# Create DataFrame
feature_importance_df = pd.DataFrame({"Symptom": X.columns, "Importance": importances})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Show top 15 symptoms
print(feature_importance_df.head(15))

N = 20
selected_features = feature_importance_df["Symptom"].head(N).tolist()
# Keep only the selected features
X_selected = X[selected_features]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Check shapes
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

X_train

"""# Logistic Regression"""

# Use Logistic Regression as base model for RFE
model = LogisticRegression(max_iter=500)
rfe = RFE(model, n_features_to_select=15)
rfe.fit(X_train, y_train)

# Get selected features
selected_features = X_train.columns[rfe.support_]
print("Selected Symptoms:", list(selected_features))

# Subset the datasets
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

"""##Logistic regression performance"""

# Optional check
if set(X_train_selected.columns) == set(X_test_selected.columns):
    print("Test cases have the correct 15 features.")
else:
    print("Test cases are missing features!")
    print("Missing Columns:", set(X_train_selected.columns) - set(X_test_selected.columns))

# Train and evaluate Logistic Regression
model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy (Logistic Regression on selected features): {accuracy:.2f}")

"""# RandomForest classifier"""

# Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Apply RFE
rfe = RFE(rf_model, n_features_to_select=15)  # Select top 20 features
rfe.fit(X_train, y_train)

# Get selected features
selected_features = X_train.columns[rfe.support_]
print("Selected Symptoms:", list(selected_features))

# Predict on test set
y_pred = rfe.predict(X_test)

# Combine results in a DataFrame
test_results = X_test.copy()
test_results["Actual Disease"] = y_test
test_results["Predicted Disease"] = y_pred

# Display some test cases
print(test_results[['Predicted Disease', 'Actual Disease']])

"""## RandomForest classifier Performance"""

# Model accuracy on test data
accuracy = rfe.score(X_test, y_test)  # This uses the internal model inside RFE
# print(f" RandomForest Model Accuracy: {accuracy:.4f}")

# Alternative way (if using predictions)
accuracy_alt = accuracy_score(y_test, y_pred)
print(f" Randomforest Accuracy: {accuracy_alt:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import pickle

# Save the trained model
with open("random_forest_rfe_model.pkl", "wb") as model_file:
    pickle.dump(rfe, model_file)

print("Model saved successfully using pickle!")

"""# Decision Tree classifier"""

# Initialize Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Apply RFE
rfe = RFE(dt_model, n_features_to_select=15)  # Select top 20 features
rfe.fit(X_train, y_train)

# Get selected features
selected_features = X_train.columns[rfe.support_]
print("Selected Symptoms:", list(selected_features))

# Predict on test set
y_pred = rfe.predict(X_test)

"""## Decision Tree moddel Performance"""

# Combine results in a DataFrame
test_results = X_test.copy()
test_results["Actual Disease"] = y_test
test_results["Predicted Disease"] = y_pred

# Display some test cases

# print(test_results[['Predicted Disease', 'Actual Disease']])
accuracy = accuracy_score(y_test, y_pred)
print(f" Decision Tree Test Accuracy: {accuracy:.2f}")

"""# Naive Bayes Classifier"""

# Apply SelectKBest with chi2
selector = SelectKBest(score_func=chi2, k=15)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get selected feature names
mask = selector.get_support()
selected_features = X_train.columns[mask]
print("Selected Symptoms:", list(selected_features))

# Train Naive Bayes on selected features
nb_model = GaussianNB()
nb_model.fit(X_train_selected, y_train)

"""## Naive Bayes model performance"""

# Predict and evaluate
y_pred = nb_model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f" Naive Bayes Test Accuracy: {accuracy:.2f}")

# Combine and display results
test_results = X_test[selected_features].copy()
test_results["Actual Disease"] = y_test.values
test_results["Predicted Disease"] = y_pred

# print(test_results[['Predicted Disease', 'Actual Disease']])

"""Support Vector Machine(SVM)"""

# Initialize SVM model with linear kernel
svm_model = SVC(kernel='linear', random_state=42)

# Apply RFE
rfe = RFE(estimator=svm_model, n_features_to_select=15)
rfe.fit(X_train, y_train)

# Get selected features
selected_features = X_train.columns[rfe.support_]
print("Selected Symptoms:", list(selected_features))

# Predict on test set using only selected features
y_pred = rfe.predict(X_test)

# Compute and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("SVM Test Accuracy:", accuracy)

"""# KNN(K-Nearest Neighbor)"""

# Step 1: Use Logistic Regression for RFE

from joblib import parallel_backend


 with parallel_backend('threading'):
     log_reg = LogisticRegression(max_iter=1000, random_state=42)
     rfe = RFE(estimator=log_reg, n_features_to_select=15)
     rfe.fit(X_train, y_train)

# Get selected features
#selected_features = X_train.columns[rfe.support_]
# print("Selected Symptoms:", list(selected_features))

# Step 2: Train KNN using selected features
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[selected_features], y_train)

# Step 3: Predict on test set using selected features
y_pred = knn.predict(X_test[selected_features])

# Step 4: Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("KNN Test Accuracy:", accuracy)


"""XGBoost"""

# Encode y labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Initialize XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Apply RFE
rfe = RFE(estimator=xgb_model, n_features_to_select=15)
rfe.fit(X_train, y_train_encoded)

# Get selected features
selected_features = X_train.columns[rfe.support_]
# print("Selected Symptoms:", list(selected_features))

# Train a model on selected features
xgb_model.fit(X_train[selected_features], y_train_encoded)

# Predict on test set
y_pred_encoded = xgb_model.predict(X_test[selected_features])

# Decode predictions back to disease names (optional)
y_pred = le.inverse_transform(y_pred_encoded)

# Accuracy (use encoded labels here)
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print("XGBoost Test Accuracy:", accuracy)

"""# Gradient Boosting Classifier"""

#Encode target labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Initialize Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)

# Apply RFE
rfe = RFE(estimator=gb_model, n_features_to_select=15)
rfe.fit(X_train, y_train_encoded)

# Get selected features
selected_features = X_train.columns[rfe.support_]
# print("Selected Symptoms:", list(selected_features))

# Train model on selected features
gb_model.fit(X_train[selected_features], y_train_encoded)

# Predict on test set
y_pred_encoded = gb_model.predict(X_test[selected_features])

# Decode predictions (optional)
y_pred = le.inverse_transform(y_pred_encoded)

# Accuracy
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print(" Gradient Boosting classifier Test Accuracy:", accuracy)

"""# Light GBM Classifier"""

# Initialize LightGBM model

lgbm_model = LGBMClassifier(random_state=42, verbosity=-1)

# Apply RFE
rfe = RFE(estimator=lgbm_model, n_features_to_select=15)
rfe.fit(X_train, y_train_encoded)

# Get selected features
selected_features = X_train.columns[rfe.support_]
# print("Selected Symptoms:", list(selected_features))

# Train model with selected features
lgbm_model.fit(X_train[selected_features], y_train_encoded)

# Predict on test set
y_pred_encoded = lgbm_model.predict(X_test[selected_features])

# Decode predictions (optional)
y_pred = le.inverse_transform(y_pred_encoded)

# Accuracy
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print("Light GBM Test Accuracy:", accuracy)