import os
import pandas as pd
from dotenv import load_dotenv
from google.cloud import dialogflow_v2 as dialogflow

# Load environment variables from .env file
load_dotenv()

# Set up Dialogflow client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
project_id = os.getenv("PROJECT_ID")
intents_client = dialogflow.IntentsClient()  # Use IntentsClient for creating intents
sessions_client = dialogflow.SessionsClient()  # Use SessionsClient for user interactions
parent = f"projects/{project_id}/agent"  # Parent path for Dialogflow agent

# Load datasets using relative paths
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
symptom_to_disease = pd.read_csv(os.path.join(base_dir, "Dataset1", "disease.csv"))  # Disease and associated symptoms
disease_description = pd.read_csv(os.path.join(base_dir, "Dataset1", "symptom_Description.csv"))  # Disease descriptions
precautions = pd.read_csv(os.path.join(base_dir, "Dataset1", "symptom_precaution.csv"))  # Precautions for each disease
symptom_severity = pd.read_csv(os.path.join(base_dir, "Dataset1", "Symptom-severity.csv"))  # Symptom severity (1-7)

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

# Function to create intents in Dialogflow
def create_intents():
    for disease, row in grouped_symptoms.iterrows():
        symptoms = row[1:]  # Extract symptoms for the disease

        # Define training phrases
        training_phrases = []
        for symptom_list in symptoms:
            for symptom in symptom_list:
                part = dialogflow.Intent.TrainingPhrase.Part(text=symptom)
                training_phrase = dialogflow.Intent.TrainingPhrase(parts=[part])
                training_phrases.append(training_phrase)

        # Check if the disease exists in disease_description
        if disease not in disease_description["Disease"].values:
            print(f"Warning: Disease '{disease}' not found in disease_description.csv. Skipping intent creation.")
            continue  # Skip to the next disease

        # Define response
        description = disease_description[disease_description["Disease"] == disease]["Description"].values[0]
        precaution_columns = [col for col in precautions.columns if col.startswith("Precaution_")]
        disease_precautions = precautions[precautions["Disease"] == disease][precaution_columns].values.flatten()
        precaution_text = ", ".join([p for p in disease_precautions if pd.notna(p)])  # Remove NaN values
        message_text = f"You might have {disease}. Description: {description}. Precautions: {precaution_text}."
        message = dialogflow.Intent.Message(text=dialogflow.Intent.Message.Text(text=[message_text]))

        # Create intent
        intent = dialogflow.Intent(
            display_name=disease,  # Use disease name as the display name
            training_phrases=training_phrases,
            messages=[message]
        )

        # Add intent to Dialogflow
        response = intents_client.create_intent(parent=parent, intent=intent)
        print(f"Intent created: {response.display_name}")

    print("All intents created successfully!")

# Function to handle user input and generate responses
def handle_user_input(user_input, session_id):
    # Extract symptoms from user input
    symptoms = extract_symptoms(user_input)
    if not symptoms:
        return "Could you describe your symptoms in more detail?"

    # Identify possible diseases
    possible_diseases = identify_diseases(symptoms)
    if len(possible_diseases) > 1:
        # Ask for additional symptoms to narrow down the disease
        next_symptoms = ask_next_symptom(possible_diseases)
        return f"Do you also have any of these symptoms: {', '.join(next_symptoms)}?"
    elif len(possible_diseases) == 1:
        # Provide disease information
        disease = possible_diseases[0]
        description, precaution_text = get_disease_info(disease)
        if check_emergency(symptoms):
            return f"You might have {disease}. {description}. Precautions: {precaution_text}. This seems serious. Please seek emergency help!"
        else:
            return f"You might have {disease}. {description}. Precautions: {precaution_text}."
    else:
        return "I couldn't identify any matching diseases. Please consult a doctor."

# Main function to interact with the chatbot
def main():
    # Create intents in Dialogflow (run this only once)
    create_intents()

    # Simulate a conversation with the chatbot
    session_id = "test-session-id"  # Unique session ID for the user
    print("Chatbot: Hello! I'm your health assistant. What symptoms are you experiencing?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        response = handle_user_input(user_input, session_id)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()