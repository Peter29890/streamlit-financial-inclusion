# Install the necessary packages
import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import your data and perform basic data exploration phase

# Load the dataset
F_I_D = pd.read_csv('Financial_inclusion_dataset.csv')

# Display the first few rows of the dataset
print(F_I_D.head())

# Display summary statistics:
print(F_I_D.info())
print(F_I_D.describe(include='all'))

# Checking the shape of the dataset:
print(F_I_D.shape)

# Handle Missing and corrupted values
# Identifying missing values:
print(F_I_D.isnull().sum())

# Define expected categories for each categorical column:
expected_categories = {
    'country': ['Rwanda', 'Tanzania', 'Kenya', 'Uganda'],
    'year': [2016, 2017, 2018],
    'bank_account': ['Yes', 'No'],
    'location_type': ['Rural', 'Urban'],
    'cellphone_access': ['Yes', 'No'],
    'gender_of_respondent': ['Female', 'Male'],
    'relationship_with_head': ['Head of Household', 'Spouse', 'Child', 'Parent', 'Other relative', 'Other non-relatives'],
    'marital_status': ['Married/Living together', 'Divorced/Separated', 'Widowed', 'Single/Never Married', 'Dont know'],
    'education_level': ['No formal education', 'Primary education', 'Secondary education', 'Tertiary education', 'Vocational/Specialised training'],
    'job_type': ['Farming and Fishing', 'Self employed', 'Formally employed Government', 'Formally employed Private', 'Informally employed', 
                 'Remittance Dependent', 'Government Dependent', 'Other Income', 'No Income', 'Dont Know/Refuse to answer']
}

# Check for unexpected or invalid values in each column:
corrupted_values = {}

for column, expected in expected_categories.items():
    unique_values = F_I_D[column].unique()
    unexpected_values = [value for value in unique_values if value not in expected]
    if unexpected_values:
        corrupted_values[column] = unexpected_values

print(corrupted_values)

# Check for valid data types:
print(F_I_D.dtypes)

# Remove duplicates if they exist
# Assuming 'F_I_D' is your DataFrame
duplicates = F_I_D[F_I_D.duplicated()]

# To check the number of duplicates
num_duplicates = duplicates.shape[0]

# Display or handle duplicates as needed
print(f"Number of duplicate rows: {num_duplicates}")
print(duplicates)

# Handle outliers if they exist
# Calculate Z-scores for numerical columns
numeric_cols = ['year', 'household_size', 'age_of_respondent']
z_scores = np.abs((F_I_D[numeric_cols] - F_I_D[numeric_cols].mean()) / F_I_D[numeric_cols].std())

# Define a Z-score threshold (e.g., 3 standard deviations)
threshold = 3

# Filter out rows with any numeric column having Z-score greater than threshold
F_I_D_Cleaned = F_I_D[(z_scores < threshold).all(axis=1)]

# Display the shape of cleaned dataset
print("Original data shape:", F_I_D.shape)
print("Cleaned data shape:", F_I_D_Cleaned.shape)

# Encode categorical features
# Importing necessary library

# Selecting categorical columns for one-hot encoding
cat_columns = ['country', 'bank_account', 'location_type', 'cellphone_access', 
               'gender_of_respondent', 'relationship_with_head', 'marital_status', 
               'education_level', 'job_type']

# Initialize OneHotEncoder without sparse argument
encoder = OneHotEncoder(drop='first')

# Fit and transform the categorical columns
encoded_cols = encoder.fit_transform(F_I_D_Cleaned[cat_columns])

# Convert to DataFrame and assign column names
encoded_cols = pd.DataFrame(encoded_cols.toarray(), columns=encoder.get_feature_names_out(cat_columns))

# Drop original categorical columns from F_I_D_Cleaned
F_I_D_Cleaned = F_I_D_Cleaned.drop(columns=cat_columns)

# Concatenate encoded columns with F_I_D_Cleaned
F_I_D_Cleaned = pd.concat([F_I_D_Cleaned, encoded_cols], axis=1)

# Displaying the first few rows of the encoded dataset
print(F_I_D_Cleaned.head())

# Save cleaned dataset to CSV
F_I_D_Cleaned.to_csv('Financial_inclusion_dataset_cleaned.csv', index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your cleaned dataset
# Assuming your cleaned data DataFrame is named `F_I_D_Cleaned`
# Ensure there are no NaN values in `bank_account_Yes`

# Check for NaN values in the entire DataFrame
print(F_I_D_Cleaned.isnull().sum())

# Ensure 'bank_account_Yes' column exists
if 'bank_account_Yes' in F_I_D_Cleaned.columns:
    # Drop rows with NaN values in 'bank_account_Yes'
    F_I_D_Cleaned.dropna(subset=['bank_account_Yes'], inplace=True)

    # Double-check if NaN values are removed
    print(F_I_D_Cleaned.isnull().sum())

    # Proceed with further processing

    # Define features and target variable
    X = F_I_D_Cleaned.drop(['year', 'uniqueid', 'bank_account_Yes'], axis=1)  # Drop 'year', 'uniqueid', and target variable
    y = F_I_D_Cleaned['bank_account_Yes']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize RandomForestClassifier
    clf = RandomForestClassifier(random_state=42)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Classification report
    print(classification_report(y_test, y_pred, target_names=['No Bank Account', 'Has Bank Account']))

else:
    print("'bank_account_Yes' column not found. Check your DataFrame columns.")

# Check for NaN values in the entire DataFrame
nan_counts = F_I_D_Cleaned.isnull().sum()

# Print columns with NaN values and their respective counts
for column, count in nan_counts.items():
    if count > 0:
        print(f"Column '{column}' has {count} NaN values.")
    else:
        print(f"Column '{column}' has no NaN values.")