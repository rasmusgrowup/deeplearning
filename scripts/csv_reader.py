import pandas as pd

# Replace with your file's path
file_path = "data/results/training_results.csv"

# Load the CSV file
df = pd.read_csv(file_path)

# Check the first few rows
print(df.head())

# Verify data types and missing values
print(df.info())