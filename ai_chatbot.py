import os
import json
import pandas as pd
from dotenv import load_dotenv
from google.cloud import dialogflow_v2 as dialogflow
from flask import Flask, request, jsonify

# Load environment variables from .env file
load_dotenv()

# Set up Dialogflow client
google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # Path to service account JSON file
project_id = os.getenv("PROJECT_ID")  # Google Cloud project ID

if not google_credentials or not project_id:
    raise ValueError("Please ensure .env file contains GOOGLE_APPLICATION_CREDENTIALS and PROJECT_ID.")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials

intents_client = dialogflow.IntentsClient()  # Use IntentsClient for creating intents
sessions_client = dialogflow.SessionsClient()  # Use SessionsClient for user interactions
parent = f"projects/{project_id}/agent"  # Parent path for Dialogflow agent

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use relative paths for data files
data_dir = os.path.join(BASE_DIR, "Dataset1")  # Directory containing datasets

symptom_to_disease_path = os.path.join(data_dir, "disease.csv")
disease_description_path = os.path.join(data_dir, "symptom_Description.csv")
precautions_path = os.path.join(data_dir, "symptom_precaution.csv")
symptom_severity_path = os.path.join(data_dir, "Symptom-severity.csv")

# Load datasets
try:
    symptom_to_disease = pd.read_csv(symptom_to_disease_path)
    disease_description = pd.read_csv(disease_description_path)
    precautions = pd.read_csv(precautions_path)
    symptom_severity = pd.read_csv(symptom_severity_path)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Dataset file not found: {e}")

# Group symptoms by disease
grouped_symptoms = symptom_to_disease.groupby("Disease").agg(lambda x: x.dropna().tolist())

# Function to extract symptoms from user input
def extract_symptoms(user_input):
    symptoms = []
    for symptom in symptom_severity["Symptom"]:
        if symptom.lower() in user_input.lower():
            symptoms.append(symptom)
    return symptoms

# Function to identify possible diseases based on symptoms
def identify_diseases(symptoms):
    possible_diseases = []
    for disease, row in grouped_symptoms.iterrows():
        disease_symptoms = row[1:]  # Extract symptoms for the disease
        if all(symptom in disease_symptoms for symptom in symptoms):
            possible_diseases.append(disease)
    return possible_diseases

# Function to ask for additional symptoms
def ask_next_symptom(possible_diseases):
    next_symptoms = set()
    for disease in possible_diseases:
        disease_symptoms = grouped_symptoms.loc[disease][1:]
        next_symptoms.update(disease_symptoms)
    return list(next_symptoms)

# Function to get disease description and precautions
def get_disease_info(disease):
    description = disease_description[disease_description["Disease"] == disease]["Description"].values[0]
    precaution_columns = [col for col in precautions.columns if col.startswith("Precaution_")]
    disease_precautions = precautions[precautions["Disease"] == disease][precaution_columns].values.flatten()
    precaution_text = ", ".join([p for p in disease_precautions if pd.notna(p)])  # Remove NaN values
    return description, precaution_text

# Function to check symptom severity
def check_emergency(symptoms):
    for symptom in symptoms:
        severity = symptom_severity[symptom_severity["Symptom"] == symptom]["Severity"].values[0]
        if severity == 7:
            return True
    return False

# Function to detect intent using Dialogflow
def detect_intent(text, session_id):
    session = sessions_client.session_path(project_id, session_id)
    text_input = dialogflow.TextInput(text=text, language_code="en")
    query_input = dialogflow.QueryInput(text=text_input)
    response = sessions_client.detect_intent(session=session, query_input=query_input)
    return response.query_result

# Flask app for handling user interactions
app = Flask(__name__)

# Store conversation state (in-memory, replace with a database for production)
conversation_state = {}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")
    session_id = data.get("session_id")

    # Detect intent using Dialogflow
    intent_response = detect_intent(user_input, session_id)
    detected_symptoms = extract_symptoms(intent_response.query_text)

    # Update conversation state
    if session_id not in conversation_state:
        conversation_state[session_id] = {"symptoms": []}
    conversation_state[session_id]["symptoms"].extend(detected_symptoms)

    # Identify possible diseases
    possible_diseases = identify_diseases(conversation_state[session_id]["symptoms"])

    # Handle conversation flow
    if len(possible_diseases) > 1:
        next_symptoms = ask_next_symptom(possible_diseases)
        response = f"Do you also have any of these symptoms: {', '.join(next_symptoms)}?"
    elif len(possible_diseases) == 1:
        disease = possible_diseases[0]
        description, precaution_text = get_disease_info(disease)
        if check_emergency(conversation_state[session_id]["symptoms"]):
            response = f"You might have {disease}. {description}. Precautions: {precaution_text}. This seems serious. Please seek emergency help!"
        else:
            response = f"You might have {disease}. {description}. Precautions: {precaution_text}."
    else:
        response = "I couldn't identify any matching diseases. Please consult a doctor."

    return jsonify({"response": response})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)