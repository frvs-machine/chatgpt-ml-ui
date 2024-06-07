import streamlit as st
import pandas as pd
import openai
import io
import os
import numpy as np
from sklearn.impute import SimpleImputer
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
if 'training_file' not in st.session_state:
    st.session_state.training_file = None
if 'testing_file' not in st.session_state:
    st.session_state.testing_file = None
if 'cleaned_training_data' not in st.session_state:
    st.session_state.cleaned_training_data = None
if 'cleaned_testing_data' not in st.session_state:
    st.session_state.cleaned_testing_data = None
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_data
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

@st.cache_data
def process_uploaded_file(uploaded_file):
    try:
        data = pd.read_excel(uploaded_file)
        cleaned_data = clean_data_with_chatgpt(data)
        return cleaned_data
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None

def run_initial_model(training_data, testing_data):
    # Create target column for training data
    training_data['Target'] = 1  # Use 1 for the training data as the lookalike target

    # Ensure all categorical variables are converted to numerical values
    combined_data = pd.concat([training_data, testing_data], ignore_index=True)
    combined_data = pd.get_dummies(combined_data)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    combined_data_imputed = imputer.fit_transform(combined_data)

    # Split combined data back into training and testing sets
    X_train = combined_data_imputed[:len(training_data), :-1]  # Exclude Target column
    y_train = combined_data_imputed[:len(training_data), -1]  # Target column
    X_test = combined_data_imputed[len(training_data):, :-1]  # Exclude Target column

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a neural network model
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X_train, y_train)

    # Predict probabilities for the test data
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Create a DataFrame with the original testing data and the predicted probabilities
    testing_data_with_scores = testing_data.copy()
    testing_data_with_scores['Lookalike_Score'] = y_pred_proba
    return model, testing_data_with_scores

def re_run_model_based_on_query(model, training_data, testing_data, query):
    # Process the query with ChatGPT
    processed_query = process_query_with_chatgpt(query)
    st.write(f"Processed Query: {processed_query}")

    # Extract city or other parameters from the processed query (this is a simple example)
    city = None
    if "Riverview" in processed_query:
        city = "Riverview"

    # Filter testing data based on the query
    if city:
        testing_data_filtered = testing_data[testing_data['City'] == city]
    else:
        testing_data_filtered = testing_data

    if testing_data_filtered.empty:
        st.error(f"No records found for the specified query: {city}.")
        return None
    else:
        # Ensure all categorical variables are converted to numerical values
        combined_data = pd.concat([training_data, testing_data_filtered], ignore_index=True)
        combined_data = pd.get_dummies(combined_data)

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        combined_data_imputed = imputer.fit_transform(combined_data)

        # Split combined data back into training and testing sets
        X_train = combined_data_imputed[:len(training_data), :-1]  # Exclude Target column
        X_test = combined_data_imputed[len(training_data):, :-1]  # Exclude Target column

        # Scale data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Predict probabilities for the test data
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Create a DataFrame with the original testing data and the predicted probabilities
        testing_data_filtered['Lookalike_Score'] = y_pred_proba
        return testing_data_filtered

# File upload for training data
uploaded_training_file = st.file_uploader("Upload a training Excel file", type=["xlsx"], key="training_file")
if uploaded_training_file:
    st.session_state.training_file = uploaded_training_file
    st.session_state.cleaned_training_data = process_uploaded_file(uploaded_training_file)

# File upload for testing data
uploaded_testing_file = st.file_uploader("Upload a testing Excel file", type=["xlsx"], key="testing_file")
if uploaded_testing_file:
    st.session_state.testing_file = uploaded_testing_file
    st.session_state.cleaned_testing_data = process_uploaded_file(uploaded_testing_file)

# Check if both files are uploaded and processed
if st.session_state.cleaned_training_data is not None and st.session_state.cleaned_testing_data is not None:
    cleaned_training_data = st.session_state.cleaned_training_data
    cleaned_testing_data = st.session_state.cleaned_testing_data

    # Run the initial model
    if st.session_state.model is None:
        st.session_state.model, initial_results = run_initial_model(cleaned_training_data, cleaned_testing_data)
        st.write("Initial Model Results:")
        st.write(initial_results.head())

    # Add a text input for user queries
    st.session_state.user_query = st.text_input(
        "Enter your query to adjust model focus (e.g., 'Run a model for lookalikes using the training data for the testing dataset with a focus for all records in Riverview'):",
        value=st.session_state.user_query
    )

    # Add an enter button
    if st.button('Re-run Model'):
        if st.session_state.model:
            updated_results = re_run_model_based_on_query(
                st.session_state.model, cleaned_training_data, cleaned_testing_data, st.session_state.user_query
            )
            if updated_results is not None:
                st.write("Updated Model Results:")
                st.write(updated_results.head())
                # Export the results to an Excel file
                try:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        updated_results.to_excel(writer, index=False, sheet_name='Sheet1')
                    processed_data = output.getvalue()
                    st.download_button(label="Download Results",
                                       data=processed_data,
                                       file_name='updated_predictions.xlsx',
                                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                    st.success("Results are ready for download.")
                except Exception as e:
                    st.error(f"Error exporting the results: {e}")
