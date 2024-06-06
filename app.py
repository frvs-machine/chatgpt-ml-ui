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
openai.api_key = os.getenv('OPENAI_API_KEY')

st.title('ChatGPT Interface with File Upload and ML Predictions')

# Function to call OpenAI API for data cleaning
def clean_data_with_chatgpt(data):
    prompt = f"Clean the following data to have columns 'First_Name', 'Last_Name', 'Address1', 'Address2', 'City', 'State', 'Zip5':\n{data.to_csv(index=False)}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1500
    )
    cleaned_data = response.choices[0].text.strip()
    return pd.read_csv(io.StringIO(cleaned_data))

# File upload for training data
uploaded_training_file = st.file_uploader("Upload a training Excel file", type=["xlsx"])

if uploaded_training_file:
    training_data = pd.read_excel(uploaded_training_file)
    st.write('Training data uploaded successfully')
    
    cleaned_training_data = clean_data_with_chatgpt(training_data)
    st.write('Cleaned Training Data:')
    st.write(cleaned_training_data)

    # File upload for testing data
    uploaded_testing_file = st.file_uploader("Upload a testing Excel file", type=["xlsx"])
    if uploaded_testing_file:
        testing_data = pd.read_excel(uploaded_testing_file)
        st.write('Testing data uploaded successfully')
        
        cleaned_testing_data = clean_data_with_chatgpt(testing_data)
        st.write('Cleaned Testing Data:')
        st.write(cleaned_testing_data)
        
        # Combine data for preprocessing
        combined_data = pd.concat([cleaned_training_data, cleaned_testing_data])
        combined_data = pd.get_dummies(combined_data)
        
        X_train = combined_data[:len(cleaned_training_data)]
        X_test = combined_data[len(cleaned_training_data):]
        y_train = training_data['target_column']  # Replace 'target_column' with the actual target column name
        
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
