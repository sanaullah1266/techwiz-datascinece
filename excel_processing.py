import pandas as pd

# Load the Excel file
df = pd.read_excel('Sorted_Cleaned_Save_Planet_Dataset_All.xlsx')

# Display the first few rows and the column names
print("First few rows of the DataFrame:")
print(df.head())

# Check how many columns are in the DataFrame
print("\nNumber of columns in the DataFrame:", len(df.columns))
print("\nColumn names:")
print(df.columns)

# Drop the specified columns if they exist
columns_to_drop = ["Data_Filenames", "Normalized_Values"]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Check the updated DataFrame structure
print("\nUpdated DataFrame:")
print(df.head())

# Make sure there are at least 15 columns before slicing
if len(df.columns) >= 15:
    # Split the data into X (first 14 columns) and y (15th column)
    x = df.iloc[:, 0:140]  # First 14 columns
    y = df.iloc[:, 14]    # 15th column

    print("\nX (first 14 columns):")
    print(x.head())

    print("\nY (15th column):")
    print(y.head())
else:
    print(f"Error: DataFrame has less than 15 columns. It has {len(df.columns)} columns.")
