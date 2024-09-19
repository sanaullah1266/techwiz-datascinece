import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset (ensure correct file path)
df = pd.read_excel('Cleaned_Save_Planet_Dataset_All.xlsx')

# Drop non-numeric columns and prepare data for modeling
X = df.drop(columns=['Data_Filenames', 'Data_Description', 'Normalized_Values'])
y = df['Normalized_Values']

# Scale the features (SVM and Random Forest may benefit from scaled input)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define hyperparameter grids for tuning
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
param_grid_svr = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'epsilon': [0.01, 0.1, 1]
}

# Initialize models with GridSearchCV for tuning
rf_model = GridSearchCV(RandomForestRegressor(), param_grid_rf, cv=3, n_jobs=-1)
svm_model = GridSearchCV(SVR(), param_grid_svr, cv=3, n_jobs=-1)

# Train the models
models = {
    'RandomForest': rf_model,
    'SVM': svm_model
}

for model_name, model in models.items():
    model.fit(X_train, y_train)

# Evaluate model performance using cross-validation
results = {}
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    cv_scores = cross_val_score(model, X_train, y_train, cv=3)  # Cross-validation
    cv_mean = np.mean(cv_scores)

    results[model_name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'CV Score (mean)': cv_mean
    }

# Display the evaluation results
results_df = pd.DataFrame(results).T
print(results_df)

# Predict using the best performing model (Random Forest in this case)
best_rf_model = models['RandomForest'].best_estimator_  # Get the best estimator from GridSearchCV
y_pred_rf = best_rf_model.predict(X_test)

# Show the predicted values along with actual values for comparison
predicted_output = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_rf
})

print(predicted_output.head())  # Display first few rows of actual vs predicted

# Scatter plot: Actual vs Predicted values for Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='blue', label='Predictions')
plt.title('Actual vs Predicted Normalized Crop Yield (Random Forest)')
plt.xlabel('Actual Normalized Values')
plt.ylabel('Predicted Normalized Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')
plt.legend()
plt.grid(True)
plt.show()

# Improved Bar chart for model performance metrics
metrics_to_plot = ['MAE', 'RMSE', 'R²', 'CV Score (mean)']
results_df[metrics_to_plot].plot(kind='bar', title='Model Performance Metrics', figsize=(10, 6), color=['skyblue', 'lightgreen', 'orange', 'purple'])
plt.ylabel('Scores')
plt.grid(axis='y')
plt.show()

# Hyperparameter Tuning Results
print(f"Best Random Forest Parameters: {models['RandomForest'].best_params_}")
print(f"Best SVM Parameters: {models['SVM'].best_params_}")
