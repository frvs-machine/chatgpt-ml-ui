import streamlit as st
import pandas as pd
import openai
import io
import os
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

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

# Function to run the model
def run_model(training_data, testing_data):
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

    # Debugging: Print shapes of training and testing sets
    st.write(f"X_train shape: {X_train.shape}")
    st.write(f"y_train shape: {y_train.shape}")
    st.write(f"X_test shape: {X_test.shape}")

    if X_test.shape[0] == 0:
        st.error("X_test is empty. Please check the data processing steps.")
        return None, None
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
        testing_data_with_scores = testing_data.copy()
        testing_data_with_scores['Lookalike_Score'] = y_pred_proba
        testing_data_with_scores['Percentile'] = pd.qcut(y_pred_proba, 100, labels=False) + 1
        return model, testing_data_with_scores

# Function to process and re-run the model based on user input
def re_run_model_based_on_query(model, training_data, testing_data, query):
    if "run for lookalikes in" in query.lower():
        emphasis_city = query.lower().replace("run for lookalikes in", "").strip()
        st.write(f"Re-running model with emphasis on {emphasis_city}")

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

        # Emphasize the city in the scoring
        emphasis_factor = 1.2  # Increase score by 20% for emphasis
        emphasis_indices = testing_data.apply(lambda row: emphasis_city in row.astype(str).str.lower().values, axis=1)
        y_pred_proba[emphasis_indices] *= emphasis_factor

        # Recalculate percentiles after emphasis
        percentiles = pd.qcut(y_pred_proba, 100, labels=False) + 1

        # Create a DataFrame with the original testing data and the predicted probabilities
        testing_data_with_scores = testing_data.copy()
        testing_data_with_scores['Lookalike_Score'] = y_pred_proba
        testing_data_with_scores['Percentile'] = percentiles

        st.write("Model re-run with emphasis on:")
        st.write(testing_data_with_scores.head())
        return testing_data_with_scores

    elif "give me results only from" in query.lower():
        filter_city = query.lower().replace("give me results only from", "").strip()
        st.write(f"Filtering results to only include {filter_city}")

        # Filter results based on the specified city
        matching_indices = testing_data.apply(lambda row: filter_city in row.astype(str).str.lower().values, axis=1)
        filtered_data = testing_data[matching_indices]
        
        if filtered_data.empty:
            st.error(f"No records found for the specified query: {filter_city}.")
        else:
            filtered_data['Lookalike_Score'] = testing_data_with_scores.loc[matching_indices, 'Lookalike_Score']
            filtered_data['Percentile'] = testing_data_with_scores.loc[matching_indices, 'Percentile']
            st.write("Filtered Results:")
            st.write(filtered_data.head())
            return filtered_data

    else:
        st.error("Invalid query format. Please use 'Run for lookalikes in [City]' or 'Give me results only from [City]'.")

# File upload for training data
uploaded_training_file = st.file_uploader("Upload a training Excel file", type=["xlsx"])

if uploaded_training_file:
    cleaned_training_data = process_uploaded_file(uploaded_training_file)

    # File upload for testing data
    uploaded_testing_file = st.file_uploader("Upload a testing Excel file", type=["xlsx"])
    if uploaded_testing_file:
        cleaned_testing_data = process_uploaded_file(uploaded_testing_file)

        if cleaned_training_data is not None and cleaned_testing_data is not None:
            # Run the initial model
            model, testing_data_with_scores = run_model(cleaned_training_data, cleaned_testing_data)
            if model is not None:
                st.write("Initial Model Results:")
                st.write(testing_data_with_scores.head())

                # Provide download button for initial results
                try:
                    initial_output = io.BytesIO()
                    with pd.ExcelWriter(initial_output, engine='xlsxwriter') as writer:
                        testing_data_with_scores.to_excel(writer, index=False, sheet_name='InitialResults')
                    initial_processed_data = initial_output.getvalue()
                    st.download_button(label="Download Initial Results",
                                       data=initial_processed_data,
                                       file_name='initial_predictions.xlsx',
                                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                except Exception as e:
                    st.error(f"Error exporting the initial results: {e}")

                # Add a text input for user queries
                user_query = st.text_input("Adjust parameters to re-run model based on user-input criteria:")

                if st.button('Re-run Model'):
                    if user_query:
                        # Process the query
                        updated_results = re_run_model_based_on_query(model, cleaned_training_data, testing_data_with_scores, user
