import os
import pandas as pd
from dotenv import load_dotenv
from google.cloud import dialogflow_v2 as dialogflow
from google.protobuf import field_mask_pb2

# Load environment variables from .env file
load_dotenv()

# Set up Dialogflow client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
project_id = os.getenv("PROJECT_ID")
intents_client = dialogflow.IntentsClient()
sessions_client = dialogflow.SessionsClient()
parent = f"projects/{project_id}/agent"  # Parent path for Dialogflow agent

# Load datasets using relative paths
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
symptom_to_disease = pd.read_csv(os.path.join(base_dir, "Dataset1", "disease.csv"))  # Disease and associated symptoms
disease_description = pd.read_csv(os.path.join(base_dir, "Dataset1", "symptom_Description.csv"))  # Disease descriptions
precautions = pd.read_csv(os.path.join(base_dir, "Dataset1", "symptom_precaution.csv"))  # Precautions for each disease
symptom_severity = pd.read_csv(os.path.join(base_dir, "Dataset1", "Symptom-severity.csv"))  # Symptom severity (1-7)

# Group symptoms by disease
grouped_symptoms = symptom_to_disease.groupby("Disease").agg(lambda x: x.dropna().tolist())

def create_intent(display_name, training_phrases, parameters=None, 
                 responses=None, input_context_names=None, 
                 output_contexts=None, webhook_enabled=True):
    """Creates a Dialogflow intent with all configurations."""
    
    # Convert training phrases to parts
    training_phrase_objects = []
    for phrase in training_phrases:
        part = dialogflow.Intent.TrainingPhrase.Part(text=phrase)
        training_phrase = dialogflow.Intent.TrainingPhrase(parts=[part])
        training_phrase_objects.append(training_phrase)
    
    # Convert parameters if provided
    intent_parameters = []
    if parameters:
        for param_name, entity_type in parameters.items():
            parameter = dialogflow.Intent.Parameter(
                display_name=param_name,
                entity_type_display_name=entity_type,
                value=f"${param_name}"
            )
            intent_parameters.append(parameter)
    
    # Convert responses
    intent_responses = []
    if responses:
        for response in responses:
            text = dialogflow.Intent.Message.Text(text=[response])
            message = dialogflow.Intent.Message(text=text)
            intent_responses.append(message)
    
    # Convert contexts
    input_contexts = []
    if input_context_names:
        for context_name in input_context_names:
            input_contexts.append(f"projects/{project_id}/agent/sessions/-/contexts/{context_name}")
    
    output_context_list = []
    if output_contexts:
        for context_name, lifespan_count in output_contexts.items():
            output_context = dialogflow.Context(
                name=f"projects/{project_id}/agent/sessions/-/contexts/{context_name}",
                lifespan_count=lifespan_count
            )
            output_context_list.append(output_context)
    
    # Create the intent
    intent = dialogflow.Intent(
        display_name=display_name,
        training_phrases=training_phrase_objects,
        parameters=intent_parameters,
        messages=intent_responses,
        input_context_names=input_contexts,
        output_contexts=output_context_list,
        webhook_state=1 if webhook_enabled else 0  # 1 = WEBHOOK_STATE_ENABLED
    )
    
    response = intents_client.create_intent(
        parent=parent,
        intent=intent
    )
    
    print(f"Created intent: {response.display_name}")
    return response

def update_intent(intent_name, updates):
    """Updates an existing intent with new configurations."""
    intents = intents_client.list_intents(parent=parent)
    intent_to_update = None
    
    for intent in intents:
        if intent.display_name == intent_name:
            intent_to_update = intent
            break
    
    if not intent_to_update:
        print(f"Intent {intent_name} not found")
        return None
    
    # Apply updates
    for field, value in updates.items():
        setattr(intent_to_update, field, value)
    
    # Specify which fields to update
    update_mask = field_mask_pb2.FieldMask()
    for field in updates.keys():
        update_mask.paths.append(field)
    
    response = intents_client.update_intent(
        intent=intent_to_update,
        update_mask=update_mask
    )
    
    print(f"Updated intent: {response.display_name}")
    return response

def create_disease_intents():
    """Creates intents for each disease based on the dataset."""
    for disease, row in grouped_symptoms.iterrows():
        symptoms = row[1:]  # Extract symptoms for the disease

        # Check if the disease exists in disease_description
        if disease not in disease_description["Disease"].values:
            print(f"Warning: Disease '{disease}' not found in disease_description.csv. Skipping intent creation.")
            continue

        # Get disease info for responses
        description = disease_description[disease_description["Disease"] == disease]["Description"].values[0]
        precaution_columns = [col for col in precautions.columns if col.startswith("Precaution_")]
        disease_precautions = precautions[precautions["Disease"] == disease][precaution_columns].values.flatten()
        precaution_text = ", ".join([p for p in disease_precautions if pd.notna(p)])
        
        # Create training phrases from symptoms
        training_phrases = []
        for symptom_list in symptoms:
            for symptom in symptom_list:
                training_phrases.append(f"I have {symptom}")
                training_phrases.append(f"I'm experiencing {symptom}")
                training_phrases.append(f"I feel {symptom}")
        
        # Create the disease intent
        create_intent(
            display_name=disease,
            training_phrases=training_phrases,
            responses=[f"You might have {disease}. Description: {description}. Precautions: {precaution_text}."],
            webhook_enabled=True
        )

def setup_conversation_intents():
    """Sets up the conversational intents for the health assistant."""
    
    # 1. Welcome Intent
    create_intent(
        display_name="Default Welcome Intent",
        training_phrases=["hello", "hi", "I need help", "start", "I'm not feeling well"],
        responses=["Hello! I'm your health assistant. What symptoms are you experiencing?"],
        webhook_enabled=False
    )
    
    # 2. General Symptoms Intent
    symptom_params = {
        "symptom": "@sys.any"
    }
    
    create_intent(
        display_name="General Symptoms",
        training_phrases=[
            "I have {symptom}",
            "I'm experiencing {symptom}",
            "I feel {symptom}",
            "My symptom is {symptom}",
            "{symptom}"
        ],
        parameters=symptom_params,
        responses=[],  # Handled by webhook
        webhook_enabled=True
    )
    
    # 3. Multiple Symptoms Intent
    multi_symptom_params = {
        "symptom1": "@sys.any",
        "symptom2": "@sys.any",
        "symptom3": "@sys.any"
    }
    
    create_intent(
        display_name="Multiple Symptoms",
        training_phrases=[
            "I have {symptom1} and {symptom2}",
            "My symptoms are {symptom1}, {symptom2} and {symptom3}",
            "I feel {symptom1} and {symptom2}",
            "{symptom1} with {symptom2}"
        ],
        parameters=multi_symptom_params,
        responses=[],  # Handled by webhook
        webhook_enabled=True
    )
    
    # 4. Follow-up Symptoms Intent
    follow_up_params = {
        "followup_symptom": "@sys.any"
    }
    
    create_intent(
        display_name="Follow-up Symptoms",
        training_phrases=[
            "yes I have {followup_symptom}",
            "I also have {followup_symptom}",
            "no just these",
            "{followup_symptom}",
            "yes, {followup_symptom}"
        ],
        parameters=follow_up_params,
        responses=[],  # Handled by webhook
        input_context_names=["awaiting_symptom_confirmation"],
        output_contexts={"awaiting_symptom_confirmation": 5},
        webhook_enabled=True
    )
    
    # 5. Emergency Intent
    create_intent(
        display_name="Emergency",
        training_phrases=[
            "it's an emergency",
            "I need help now",
            "severe pain",
            "call ambulance",
            "this is urgent",
            "I can't breathe",
            "chest pain"
        ],
        responses=["Please call emergency services immediately! This seems serious."],
        webhook_enabled=True
    )
    
    # 6. No More Symptoms Intent
    create_intent(
        display_name="No More Symptoms",
        training_phrases=[
            "no that's all",
            "no more symptoms",
            "just those",
            "that's it",
            "no others"
        ],
        responses=[],  # Handled by webhook
        input_context_names=["awaiting_symptom_confirmation"],
        webhook_enabled=True
    )
    
    # 7. Update Fallback Intent
    update_intent(
        "Default Fallback Intent",
        {
            "messages": [dialogflow.Intent.Message(
                text=dialogflow.Intent.Message.Text(text=["I'm not sure how to help with that. Could you describe your symptoms?"])
            )],
            "webhook_state": 1  # Enable webhook
        }
    )

def setup_all_intents():
    """Configures all required intents for the health assistant."""
    print("Setting up conversation intents...")
    setup_conversation_intents()
    
    print("Creating disease-specific intents...")
    create_disease_intents()
    
    print("Dialogflow intents configuration completed!")

if __name__ == "__main__":
    setup_all_intents()