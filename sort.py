import pandas as pd

# Load the Excel file
file_path = 'Cleaned_Save_Planet_Dataset_All.xlsx'  # Updated file name

try:
    df = pd.read_excel(file_path)
    print("Columns in the dataset:", df.columns)  # Print column names to verify
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the Excel file: {e}")
    exit()

# Check if 'Data_Values' column exists
if 'Data_Values' not in df.columns:
    print("Error: The column 'Data_Values' does not exist in the dataset.")
    exit()

# Sorting by 'Data_Values' in ascending order
df_sorted = df.sort_values(by='Data_Values', ascending=True)

# Display the sorted data
print(df_sorted.head())

# Save the sorted data to a new Excel file
try:
    df_sorted.to_excel('Sorted_Cleaned_Save_Planet_Dataset_All.xlsx', index=False)  # Updated output file name
    print("The sorted data has been saved to 'Sorted_Cleaned_Save_Planet_Dataset_All.xlsx'.")
except Exception as e:
    print(f"An error occurred while saving the Excel file: {e}")
