import streamlit as st
import pandas as pd
import openai
import io
import os
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Set your OpenAI API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.error("No API key provided. Please set the OPENAI_API_KEY environment variable.")
else:
    openai.api_key = api_key

st.title('ChatGPT Interface with File Upload and ML Predictions')

# Initialize session state
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""
if 'training_file' not in st.session_state:
    st.session_state.training_file = None
if 'testing_file' not in st.session_state:
    st.session_state.testing_file = None

# Add a text input for user queries
st.session_state.user_query = st.text_input(
    "Enter your query to adjust model focus (e.g., 'Run a model for lookalikes using the training data for the testing dataset with a focus for all records in Riverview'):",
    value=st.session_state.user_query
)

# Add an enter button
if st.button('Enter'):
    st.session_state.query_submitted = True

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

def process_query_with_chatgpt(query):
    prompt = (
        f"Process this query to adjust model focus for a dataset: {query}. "
        "Extract relevant information such as city names, and any other specific parameters mentioned."
    )
    messages = [
        {"role": "system", "content": "You are a data processing assistant."},
        {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500
    )
    processed_query = response['choices'][0]['message']['content']
    return processed_query

# Handle the query submission and file uploads
if st.session_state.get('query_submitted'):
    processed_query = process_query_with_chatgpt(st.session_state.user_query)
    st.write(f"Processed Query: {processed_query}")  # Debug print statement

    # Extract city or other parameters from the processed query (this is a simple example)
    city = None
    if "Riverview" in processed_query:
        city = "Riverview"

    # File upload for training data
    st.session_state.training_file = st.file_uploader("Upload a training Excel file", type=["xlsx"], key="training_file")

    if st.session_state.training_file:
        cleaned_training_data = process_uploaded_file(st.session_state.training_file)

        # File upload for testing data
        st.session_state.testing_file = st.file_uploader("Upload a testing Excel file", type=["xlsx"], key="testing_file")
        if st.session_state.testing_file:
            cleaned_testing_data = process_uploaded_file(st.session_state.testing_file)

            if cleaned_training_data is not None and cleaned_testing_data is not None:
                # Create target column for training data
                cleaned_training_data['Target'] = 1  # Use 1 for the training data as the lookalike target

                # Debugging: Check if 'Target' column is added
                st.write('Training Data with Target:')
                st.write(cleaned_training_data.head())
                st.write(cleaned_training_data.columns)  # Print columns for debugging

                # Filter testing data based on the query
                if city:
                    testing_data_filtered = cleaned_testing_data[cleaned_testing_data['City'] == city]
                else:
                    testing_data_filtered = cleaned_testing_data

                if testing_data_filtered.empty:
                    st.error(f"No records found for the specified query: {city}.")
                else:
                    # Ensure all categorical variables are converted to numerical values
                    combined_data = pd.concat([cleaned_training_data, testing_data_filtered], ignore_index=True)
                    combined_data = pd.get_dummies(combined_data)

                    # Handle missing values
                    imputer = SimpleImputer(strategy='mean')
                    combined_data_imputed = imputer.fit_transform(combined_data)

                    # Split combined data back into training and testing sets
                    X_train = combined_data_imputed[:len(cleaned_training_data), :-1]  # Exclude Target column
                    y_train = combined_data_imputed[:len(cleaned_training_data), -1]  # Target column
                    X_test = combined_data_imputed[len(cleaned_training_data):, :-1]  # Exclude Target column

                    # Debugging: Print shapes of training and testing sets
                    st.write(f"X_train shape: {X_train.shape}")
                    st.write(f"y_train shape: {y_train.shape}")
                    st.write(f"X_test shape: {X_test.shape}")

                    if X_test.shape[0] == 0:
                        st.error("X_test is empty. Please check the data processing steps.")
                    else:
                        # Scale data
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)

                        # Train a neural network model
                        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
                        model.fit(X_train, y_train)

                        # Predict probabilities for the test data
                        y_pred_proba = model.predict_proba(X_test)[:, 1]

                        # Calculate percentiles
                        percentiles = np.percentile(y_pred_proba, np.arange(100))

                        # Create a DataFrame with the original testing data and the predicted probabilities
                        testing_data_with_scores = testing_data_filtered.copy()
                        testing_data_with_scores['Lookalike_Score'] = y_pred_proba
                        testing_data_with_scores['Percentile'] = [np.sum(y_pred_proba <= x) for x in y_pred_proba]

                        # Export the results to an Excel file
                        try:
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                testing_data_with_scores.to_excel(writer, index=False, sheet_name='Sheet1')
                            processed_data = output.getvalue()
                            st.download_button(label="Download Results",
                                               data=processed_data,
                                               file_name='predictions.xlsx',
                                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                            st.success("Results are ready for download.")
                        except Exception as e:
                            st.error(f"Error exporting the results: {e}")
