# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import joblib  # To save and load the model

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

# Handle Missing and corrupted values
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

# Remove duplicates if they exist
F_I_D.drop_duplicates(inplace=True)

# Drop the uniqueid column as it is not useful for prediction
F_I_D = F_I_D.drop(columns=['uniqueid'])

# Handle outliers if they exist
numeric_cols = ['year', 'household_size', 'age_of_respondent']
z_scores = np.abs((F_I_D[numeric_cols] - F_I_D[numeric_cols].mean()) / F_I_D[numeric_cols].std())
threshold = 3
F_I_D_Cleaned = F_I_D[(z_scores < threshold).all(axis=1)]

# Encode categorical features
cat_columns = ['country', 'location_type', 'cellphone_access', 
               'gender_of_respondent', 'relationship_with_head', 'marital_status', 
               'education_level', 'job_type']

# Initialize OneHotEncoder without sparse argument
encoder = OneHotEncoder(drop='first')
encoded_cols = encoder.fit_transform(F_I_D_Cleaned[cat_columns])
encoded_cols = pd.DataFrame(encoded_cols.toarray(), columns=encoder.get_feature_names_out(cat_columns))
F_I_D_Cleaned = F_I_D_Cleaned.drop(columns=cat_columns)
F_I_D_Cleaned = pd.concat([F_I_D_Cleaned, encoded_cols], axis=1)

# Separate features and target variable
X = F_I_D_Cleaned.drop(columns=['bank_account'])
y = F_I_D_Cleaned['bank_account'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and encoder
joblib.dump(model, 'financial_inclusion_model.pkl')
joblib.dump(encoder, 'encoder.pkl')

# Streamlit application
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

# Load the model and encoder
model = joblib.load('financial_inclusion_model.pkl')
encoder = joblib.load('encoder.pkl')

# Define the Streamlit app
def main():
    st.title("Financial Inclusion Prediction")
    
    # Input fields
    country = st.selectbox("Country", ['Rwanda', 'Tanzania', 'Kenya', 'Uganda'])
    year = st.selectbox("Year", [2016, 2017, 2018])
    location_type = st.selectbox("Location Type", ['Rural', 'Urban'])
    cellphone_access = st.selectbox("Cellphone Access", ['Yes', 'No'])
    household_size = st.slider("Household Size", 1, 20, 1)
    age_of_respondent = st.slider("Age of Respondent", 16, 100, 30)
    gender_of_respondent = st.selectbox("Gender", ['Female', 'Male'])
    relationship_with_head = st.selectbox("Relationship with Head", ['Head of Household', 'Spouse', 'Child', 'Parent', 'Other relative', 'Other non-relatives'])
    marital_status = st.selectbox("Marital Status", ['Married/Living together', 'Divorced/Separated', 'Widowed', 'Single/Never Married', 'Dont know'])
    education_level = st.selectbox("Education Level", ['No formal education', 'Primary education', 'Secondary education', 'Tertiary education', 'Vocational/Specialised training'])
    job_type = st.selectbox("Job Type", ['Farming and Fishing', 'Self employed', 'Formally employed Government', 'Formally employed Private', 'Informally employed', 
                                         'Remittance Dependent', 'Government Dependent', 'Other Income', 'No Income', 'Dont Know/Refuse to answer'])

    # Prepare input data
    input_data = pd.DataFrame({
        'country': [country], 
        'year': [year], 
        'location_type': [location_type], 
        'cellphone_access': [cellphone_access], 
        'household_size': [household_size], 
        'age_of_respondent': [age_of_respondent], 
        'gender_of_respondent': [gender_of_respondent], 
        'relationship_with_head': [relationship_with_head], 
        'marital_status': [marital_status], 
        'education_level': [education_level], 
        'job_type': [job_type]
    })

    # Encode input data
    encoded_input = encoder.transform(input_data[cat_columns])
    encoded_input_df = pd.DataFrame(encoded_input.toarray(), columns=encoder.get_feature_names_out(cat_columns))
    input_data = input_data.drop(columns=cat_columns)
    input_data = pd.concat([input_data, encoded_input_df], axis=1)

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("The person is likely to have a bank account.")
        else:
            st.warning("The person is unlikely to have a bank account.")

if __name__ == '__main__':
    main()
