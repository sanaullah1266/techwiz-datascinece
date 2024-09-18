import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the Excel file
file_path = 'Save Planet Dataset.xlsx'
df = pd.read_excel(file_path)

# Rename columns for clarity
df.rename(columns={
    'Data filenames': 'Data_Filenames',
    'Example': 'Data_Values',
    'Description': 'Data_Description'
}, inplace=True)

# Preview the first 5 rows of the data
print(df.head())

# Get information about the data types and missing values
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Remove duplicates
df_cleaned = df.drop_duplicates()

# Identify numeric columns (try to convert 'Data_Values' to numeric)
df_cleaned['Data_Values'] = pd.to_numeric(df_cleaned['Data_Values'], errors='coerce')

# Re-identify numeric columns after conversion
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns

# Handle missing data
# Numeric columns: Fill missing values with the mean
df_cleaned[numeric_cols] = df_cleaned[numeric_cols].apply(lambda col: col.fillna(col.mean()))

# Categorical columns: Fill missing values with the mode
df_cleaned[categorical_cols] = df_cleaned[categorical_cols].apply(lambda col: col.fillna(col.mode()[0]))

# Standardize numeric columns (if there are any numeric columns)
if len(numeric_cols) > 0:
    scaler = StandardScaler()
    df_cleaned[numeric_cols] = scaler.fit_transform(df_cleaned[numeric_cols])

# Normalize numeric columns using MinMaxScaler
if len(numeric_cols) > 0:
    min_max_scaler = MinMaxScaler()
    df_cleaned['Normalized_Values'] = min_max_scaler.fit_transform(df_cleaned[numeric_cols])

# Advanced outlier detection using both Z-score and IQR
def remove_outliers(df, col):
    # Z-score method (remove points with z-score > 3)
    df = df[(np.abs(stats.zscore(df[col])) < 3)]
    
    # IQR method (remove points beyond 1.5 times IQR)
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
    
    return df

# Remove outliers in numeric columns
for col in numeric_cols:
    df_cleaned = remove_outliers(df_cleaned, col)

# Save cleaned dataset
df_cleaned.to_excel('Cleaned_Save_Planet_Dataset_All.xlsx', index=False)

print("Data cleaning completed and saved.")
