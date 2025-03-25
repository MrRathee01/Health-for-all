from flask import Flask, request, jsonify
from google.cloud import translate_v2 as translate
from google.cloud import dialogflow_v2 as dialogflow
import os
import pandas as pd
from dotenv import load_dotenv
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = int(os.environ.get('PORT', 8080))

load_dotenv()

# Initialize clients
translate_client = translate.Client()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
dialogflow_client = dialogflow.SessionsClient()

# Load datasets
base_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(base_dir, "Dataset1")

# Verify and load datasets
try:
    symptom_to_disease = pd.read_csv(os.path.join(DATASET_PATH, "disease.csv"))
    disease_description = pd.read_csv(os.path.join(DATASET_PATH, "symptom_Description.csv"))
    precautions = pd.read_csv(os.path.join(DATASET_PATH, "symptom_precaution.csv"))
    symptom_severity = pd.read_csv(os.path.join(DATASET_PATH, "Symptom-severity.csv"))
except FileNotFoundError as e:
    logger.error(f"Dataset loading failed: {str(e)}")
    raise

# Group symptoms by disease
grouped_symptoms = symptom_to_disease.groupby("Disease").agg(lambda x: x.dropna().tolist())

# Supported Indian languages
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'bn': 'Bengali',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'pa': 'Punjabi'
}

# Symptom synonyms mapping
SYMPTOM_SYNONYMS = {
    'fever': ['feverish', 'high temperature', 'hot'],
    'headache': ['head pain', 'migraine'],
    'nausea': ['sick', 'queasy'],
    # Add more mappings
}

def detect_language(text):
    """Detect language of input text"""
    try:
        result = translate_client.detect_language(text)
        return result['language']
    except Exception as e:
        logger.error(f"Language detection failed: {str(e)}")
        return 'en'

def translate_text(text, target_language):
    """Translate text to target language"""
    try:
        if target_language == 'en':
            return text
        result = translate_client.translate(text, target_language=target_language)
        return result['translatedText']
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return text

def normalize_symptom(symptom_text, lang='en'):
    """Normalize symptom text with translation support"""
    if lang != 'en':
        symptom_text = translate_text(symptom_text, 'en')
    
    symptom_text = symptom_text.lower().strip()
    for symptom in symptom_severity["Symptom"]:
        if symptom.lower() == symptom_text:
            return symptom
    for canonical, synonyms in SYMPTOM_SYNONYMS.items():
        if symptom_text in synonyms or symptom_text == canonical:
            return canonical
    return None

def extract_symptoms(text, lang='en'):
    """Extract symptoms with multilingual support"""
    if lang != 'en':
        text = translate_text(text, 'en')
    
    symptoms = []
    for symptom in symptom_severity["Symptom"]:
        if symptom.lower() in text.lower():
            symptoms.append(symptom)
    
    found_synonyms = set()
    for canonical, synonyms in SYMPTOM_SYNONYMS.items():
        for synonym in synonyms:
            if synonym in text.lower() and canonical not in found_synonyms:
                symptoms.append(canonical)
                found_synonyms.add(canonical)
    
    return list(set(symptoms))

def identify_diseases(symptoms):
    """Identify possible diseases based on symptoms"""
    possible_diseases = []
    for disease, row in grouped_symptoms.iterrows():
        disease_symptoms = row[1:]  # Skip disease name column
        if all(symptom in disease_symptoms for symptom in symptoms):
            possible_diseases.append(disease)
    return possible_diseases

def ask_next_symptom(possible_diseases, target_lang='en'):
    """Get next symptoms to ask (with translation)"""
    next_symptoms = set()
    for disease in possible_diseases:
        disease_symptoms = grouped_symptoms.loc[disease][1:]
        next_symptoms.update(disease_symptoms)
    
    symptoms_list = list(next_symptoms)
    if target_lang != 'en':
        symptoms_list = [translate_text(s, target_lang) for s in symptoms_list]
    return symptoms_list

def get_disease_info(disease, target_lang='en'):
    """Get description and precautions (with translation)"""
    try:
        description = disease_description[disease_description["Disease"] == disease]["Description"].values[0]
        precaution_columns = [col for col in precautions.columns if col.startswith("Precaution_")]
        disease_precautions = precautions[precautions["Disease"] == disease][precaution_columns].values.flatten()
        precaution_text = ", ".join([p for p in disease_precautions if pd.notna(p)])
        
        if target_lang != 'en':
            description = translate_text(description, target_lang)
            precaution_text = translate_text(precaution_text, target_lang)
        
        return description, precaution_text
    except Exception as e:
        logger.error(f"Error getting disease info: {str(e)}")
        default_desc = translate_text("No description available", target_lang)
        default_prec = translate_text("No precautions available", target_lang)
        return default_desc, default_prec

def check_emergency(symptoms):
    """Check if symptoms indicate emergency"""
    for symptom in symptoms:
        severity = symptom_severity[symptom_severity["Symptom"] == symptom]["Severity"].values[0]
        if severity == 7:
            return True
    return False

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        req = request.get_json(silent=True, force=True)
        logger.info(f"Incoming request: {req}")
        
        # Extract language from request or detect
        user_input = req.get("queryResult", {}).get("queryText", "")
        lang = req.get("queryResult", {}).get("languageCode", detect_language(user_input))
        
        # Validate language
        if lang not in SUPPORTED_LANGUAGES:
            lang = 'en'
        
        # Process with Dialogflow
        session_id = req["session"].split("/")[-1]
        session_path = dialogflow_client.session_path(os.getenv("PROJECT_ID"), session_id)
        
        text_input = dialogflow.TextInput(text=user_input, language_code=lang)
        query_input = dialogflow.QueryInput(text=text_input)
        dialogflow_response = dialogflow_client.detect_intent(
            session=session_path, 
            query_input=query_input
        )
        
        # Get parameters from Dialogflow
        parameters = dialogflow_response.query_result.parameters
        symptoms = []
        
        # Extract symptoms from parameters
        for param, value in parameters.items():
            if "symptom" in param.lower():
                if isinstance(value, list):
                    symptoms.extend([normalize_symptom(s, lang) for s in value if s])
                elif isinstance(value, str):
                    symptoms.extend(extract_symptoms(value, lang))
        
        symptoms = list(set([s for s in symptoms if s]))
        
        # Main processing logic
        intent_name = dialogflow_response.query_result.intent.display_name
        response_text = ""
        
        if intent_name in ["General Symptoms", "Multiple Symptoms"]:
            if not symptoms:
                response_text = translate_text("Could you describe your symptoms in more detail?", lang)
            else:
                possible_diseases = identify_diseases(symptoms)
                if len(possible_diseases) > 1:
                    next_symptoms = ask_next_symptom(possible_diseases, lang)
                    response_text = translate_text(
                        f"Do you also have any of these symptoms: {', '.join(next_symptoms)}?", 
                        lang
                    )
                elif len(possible_diseases) == 1:
                    disease = possible_diseases[0]
                    description, precaution_text = get_disease_info(disease, lang)
                    if check_emergency(symptoms):
                        emergency_msg = translate_text(
                            "EMERGENCY: Please seek immediate medical attention!", 
                            lang
                        )
                        response_text = f"{emergency_msg} {translate_text('You might have', lang)} {disease}. {description}. {translate_text('Precautions:', lang)} {precaution_text}"
                    else:
                        response_text = f"{translate_text('You might have', lang)} {disease}. {description}. {translate_text('Precautions:', lang)} {precaution_text}"
                else:
                    response_text = translate_text(
                        "I couldn't identify any matching diseases. Please consult a doctor.", 
                        lang
                    )
        
        elif intent_name == "Follow-up Symptoms":
            # Handle follow-up logic with context
            pass  # Similar structure as above with context handling
        
        elif intent_name == "No More Symptoms":
            # Handle no more symptoms case
            pass  # Similar structure as above
        
        else:
            response_text = translate_text(
                "I didn't understand that. Could you describe your symptoms?", 
                lang
            )
        
        # Prepare response
        response = {
            "fulfillmentText": response_text,
            "payload": {
                "google": {
                    "expectUserResponse": True,
                    "richResponse": {
                        "items": [
                            {
                                "simpleResponse": {
                                    "textToSpeech": response_text,
                                    "displayText": response_text
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in webhook: {str(e)}")
        error_msg = translate_text("Sorry, I encountered an error. Please try again.", "en")
        return jsonify({"fulfillmentText": error_msg})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)