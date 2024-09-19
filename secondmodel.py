# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, mean_squared_error, 
                             mean_absolute_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris, load_boston
import numpy as np

# -------------------- Classification: Iris Dataset -------------------- #

# Load Iris dataset
iris_data = load_iris()
X_iris, y_iris = iris_data.data, iris_data.target

# Split the data
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

# Train a RandomForest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_iris, y_train_iris)

# Predict on the test data
y_pred_iris = clf.predict(X_test_iris)

# Calculate classification metrics
accuracy = accuracy_score(y_test_iris, y_pred_iris)
precision = precision_score(y_test_iris, y_pred_iris, average='macro')
recall = recall_score(y_test_iris, y_pred_iris, average='macro')
f1 = f1_score(y_test_iris, y_pred_iris, average='macro')

# Prepare classification metrics for plotting
classification_metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}

# Plot classification metrics
plt.figure(figsize=(8, 6))
sns.barplot(x=list(classification_metrics.keys()), y=list(classification_metrics.values()), palette='Blues_d')
plt.title("Classification Metrics (Iris Dataset)")
plt.ylabel('Score')
plt.show()

# -------------------- Regression: Boston Housing Dataset -------------------- #

# Load the Boston Housing dataset
boston_data = load_boston()
X_boston, y_boston = boston_data.data, boston_data.target

# Split the data
X_train_boston, X_test_boston, y_train_boston, y_test_boston = train_test_split(
    X_boston, y_boston, test_size=0.2, random_state=42
)

# Train a Linear Regression model
reg = LinearRegression()
reg.fit(X_train_boston, y_train_boston)

# Predict on the test data
y_pred_boston = reg.predict(X_test_boston)

# Calculate regression metrics
mse = mean_squared_error(y_test_boston, y_pred_boston)
mae = mean_absolute_error(y_test_boston, y_pred_boston)
r2 = r2_score(y_test_boston, y_pred_boston)
rmse = np.sqrt(mse)

# Prepare regression metrics for plotting
regression_metrics = {
    'MSE': mse,
    'MAE': mae,
    'R-squared': r2,
    'RMSE': rmse
}

# Plot regression metrics
plt.figure(figsize=(8, 6))
sns.barplot(x=list(regression_metrics.keys()), y=list(regression_metrics.values()), palette='Greens_d')
plt.title("Regression Metrics (Boston Housing Dataset)")
plt.ylabel('Score')
plt.show()