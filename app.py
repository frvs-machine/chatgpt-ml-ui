import streamlit as st
import pandas as pd
import openai
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Set your OpenAI API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')
st.write(f"API Key: {api_key}")  # Debug print statement
if api_key is None:
    st.error("No API key provided. Please set the OPENAI_API_KEY environment variable.")
else:
    openai.api_key = api_key

st.title('ChatGPT Interface with File Upload and ML Predictions')

# Function to call OpenAI API for data cleaning
def clean_data_with_chatgpt(dataframe):
    prompt = (
        "Clean this data and format it into the following columns: First_Name, Last_Name, "
        "Address1, Address2, City, State, Zip5. Ensure the data matches the format."
    )

    messages = [
        {"role": "system", "content": "You are a data cleaning assistant."},
        {"role": "user", "content": prompt},
        {"role": "user", "content": dataframe.to_csv(index=False)}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1500
    )

    cleaned_data_csv = response['choices'][0]['message']['content']
    cleaned_data = pd.read_csv(io.StringIO(cleaned_data_csv))
    return cleaned_data

# Function to read and clean the uploaded file
def process_uploaded_file(uploaded_file):
    try:
        data = pd.read_excel(uploaded_file)
        st.write('File uploaded successfully')
        st.write(data.head())  # Display the first few rows for debugging
        
        cleaned_data = clean_data_with_chatgpt(data)
        st.write('Cleaned Data:')
        st.write(cleaned_data)
        return cleaned_data
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None

# File upload for training data
uploaded_training_file = st.file_uploader("Upload a training Excel file", type=["xlsx"])

if uploaded_training_file:
    cleaned_training_data = process_uploaded_file(uploaded_training_file)

    # File upload for testing data
    uploaded_testing_file = st.file_uploader("Upload a testing Excel file", type=["xlsx"])
    if uploaded_testing_file:
        cleaned_testing_data = process_uploaded_file(uploaded_testing_file)

        if cleaned_training_data is not None and cleaned_testing_data is not None:
            # Create target columns
            cleaned_training_data['Target'] = 'Yes'
            cleaned_testing_data['Target'] = 'No'

            # Combine data for preprocessing
            combined_data = pd.concat([cleaned_training_data, cleaned_testing_data])
            combined_data = pd.get_dummies(combined_data)

            X = combined_data.drop(columns=['Target'])
            y = combined_data['Target']

            # Split the combined data back into training and testing sets
            X_train = X[combined_data.index < len(cleaned_training_data)]
            X_test = X[combined_data.index >= len(cleaned_training_data)]
            y_train = y[combined_data.index < len(cleaned_training_data)]
            y_test = y[combined_data.index >= len(cleaned_training_data)]

            # Scale data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train a neural network model
            model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
            model.fit(X_train, y_train)

            # Predict and score the test data
            y_pred = model.predict(X_test)
            st.write('Predictions for the testing data:')
            st.write(y_pred)

            # Create a DataFrame with predictions
            results = pd.DataFrame(X_test, columns=X.columns)
            results['Predictions'] = y_pred
            results['Actual'] = y_test.values

            # Export the results to an Excel file
            try:
                results_file = 'predictions.xlsx'
                results.to_excel(results_file, index=False)
                st.success(f'Results have been exported to {results_file}')
            except Exception as e:
                st.error(f"Error exporting the results: {e}")
