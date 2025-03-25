from flask import Flask, request, jsonify
from google.cloud import translate_v2 as translate
from google.cloud import dialogflow_v2 as dialogflow
import os
import pandas as pd
from dotenv import load_dotenv
import logging
import re

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

try:
    symptom_to_disease = pd.read_csv(os.path.join(DATASET_PATH, "disease.csv"))
    disease_description = pd.read_csv(os.path.join(DATASET_PATH, "symptom_Description.csv"))
    precautions = pd.read_csv(os.path.join(DATASET_PATH, "symptom_precaution.csv"))
    symptom_severity = pd.read_csv(os.path.join(DATASET_PATH, "Symptom-severity.csv"))
    
    # Preprocess symptoms
    symptom_severity['Symptom'] = symptom_severity['Symptom'].str.lower()
    symptom_list = symptom_severity['Symptom'].tolist()
    
except Exception as e:
    logger.error(f"Dataset loading failed: {str(e)}")
    raise

# Group symptoms by disease
grouped_symptoms = symptom_to_disease.groupby("Disease").agg(lambda x: x.dropna().tolist())

# Enhanced symptom variations mapping
SYMPTOM_VARIATIONS = {
    'fever': ['feverish', 'high temperature', 'hot', 'pyrexia', 'burning up'],
    'headache': ['head pain', 'migraine', 'cephalalgia', 'head throbbing'],
    'nausea': ['sick', 'queasy', 'nauseous', 'upset stomach'],
    'fatigue': ['tiredness', 'exhaustion', 'lethargy', 'weariness'],
    'dizziness': ['lightheaded', 'vertigo', 'unsteady', 'woozy'],
    'cough': ['hacking cough', 'dry cough', 'tussis', 'coughing'],
    'sore throat': ['throat pain', 'pharyngitis', 'scratchy throat'],
    'chills': ['shivering', 'rigor', 'goosebumps', 'shuddering']
}

# Emergency keywords
EMERGENCY_KEYWORDS = [
    'emergency', 'urgent', 'critical', 'severe pain',
    'can\'t breathe', 'chest pain', 'unconscious', 'bleeding heavily'
]

def detect_language(text):
    """Robust language detection"""
    try:
        if not text or len(text.strip()) < 3:
            return 'en'
        result = translate_client.detect_language(text)
        return result['language'] if result['language'] in SUPPORTED_LANGUAGES else 'en'
    except Exception as e:
        logger.warning(f"Language detection failed: {str(e)}")
        return 'en'

def translate_text(text, target_language):
    """Safe translation with fallback"""
    try:
        if not text or target_language == 'en':
            return text
        return translate_client.translate(text, target_language=target_language)['translatedText']
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return text

def normalize_symptom(symptom_text, lang='en'):
    """Advanced symptom normalization"""
    if not symptom_text:
        return None
    
    # Translate if needed
    if lang != 'en':
        symptom_text = translate_text(symptom_text, 'en')
    
    symptom_text = re.sub(r'[^\w\s]', '', symptom_text.lower().strip())
    
    # 1. Check exact matches
    for symptom in symptom_list:
        if symptom == symptom_text:
            return symptom
    
    # 2. Check variations
    for canonical, variations in SYMPTOM_VARIATIONS.items():
        if symptom_text in variations or symptom_text == canonical:
            return canonical
    
    # 3. Partial matching
    for symptom in symptom_list:
        if symptom in symptom_text or symptom_text in symptom:
            return symptom
    
    return None

def extract_symptoms(text, lang='en'):
    """Comprehensive symptom extraction"""
    if not text:
        return []
    
    if lang != 'en':
        text = translate_text(text, 'en')
    
    text = text.lower()
    symptoms_found = set()
    
    # Check for multi-word symptoms first
    for symptom in symptom_list:
        if re.search(rf'\b{re.escape(symptom)}\b', text):
            symptoms_found.add(symptom)
    
    # Check variations
    for canonical, variations in SYMPTOM_VARIATIONS.items():
        for variation in variations:
            if re.search(rf'\b{re.escape(variation)}\b', text):
                symptoms_found.add(canonical)
    
    return list(symptoms_found)

def identify_diseases(symptoms):
    """Disease identification with scoring"""
    disease_scores = {}
    for disease, row in grouped_symptoms.iterrows():
        disease_symptoms = [s.lower() for s in row[1:] if isinstance(s, str)]
        match_count = sum(1 for symptom in symptoms if symptom.lower() in disease_symptoms)
        if match_count > 0:
            disease_scores[disease] = match_count / len(disease_symptoms)
    return sorted(disease_scores.keys(), key=lambda x: disease_scores[x], reverse=True)

def check_emergency(symptoms, text):
    """Comprehensive emergency detection"""
    text_lower = text.lower()
    
    # Check emergency keywords
    if any(keyword in text_lower for keyword in EMERGENCY_KEYWORDS):
        return True
    
    # Check symptom severity
    for symptom in symptoms:
        severity = symptom_severity[symptom_severity["Symptom"] == symptom]["Severity"].values[0]
        if severity >= 6:
            return True
    
    return False

def get_disease_info(disease, lang='en'):
    """Well-formatted disease information"""
    try:
        description = disease_description[disease_description["Disease"] == disease]["Description"].values[0]
        precaution_cols = [col for col in precautions.columns if col.startswith("Precaution_")]
        precautions_list = [p for p in precautions[precautions["Disease"] == disease][precaution_cols].values.flatten() if pd.notna(p)]
        
        # Format with bullet points
        precaution_text = "\n• " + "\n• ".join(precautions_list)
        
        if lang != 'en':
            description = translate_text(description, lang)
            precaution_text = translate_text(precaution_text, lang)
        
        return description, precaution_text
    except Exception as e:
        logger.error(f"Error getting disease info: {str(e)}")
        return None, None

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        req = request.get_json(silent=True, force=True)
        logger.info(f"Request received: {req}")
        
        # Extract basic information
        user_input = req.get("queryResult", {}).get("queryText", "")
        lang = req.get("queryResult", {}).get("languageCode", detect_language(user_input))
        lang = lang if lang in SUPPORTED_LANGUAGES else 'en'
        intent_name = req.get("queryResult", {}).get("intent", {}).get("displayName", "")
        
        # Extract symptoms
        parameters = req.get("queryResult", {}).get("parameters", {})
        symptoms = []
        for param, value in parameters.items():
            if "symptom" in param.lower():
                if isinstance(value, list):
                    symptoms.extend([normalize_symptom(s, lang) for s in value if s])
                elif isinstance(value, str):
                    symptoms.extend(extract_symptoms(value, lang))
        symptoms = list(set([s for s in symptoms if s]))
        
        # Main response logic
        response_text = ""
        if intent_name in ["General Symptoms", "Multiple Symptoms", "Follow-up Symptoms"]:
            if not symptoms:
                response_text = translate_text("Please describe your symptoms in more detail.", lang)
            else:
                is_emergency = check_emergency(symptoms, user_input)
                possible_diseases = identify_diseases(symptoms)
                
                if is_emergency:
                    response_text = translate_text(
                        "🚨 EMERGENCY: Please seek immediate medical attention!", 
                        lang
                    )
                elif possible_diseases:
                    disease = possible_diseases[0]
                    description, precautions = get_disease_info(disease, lang)
                    
                    if description and precautions:
                        response_text = (
                            f"{translate_text('Possible diagnosis:', lang)} {disease}\n"
                            f"{translate_text('Description:', lang)} {description}\n"
                            f"{translate_text('Recommended actions:', lang)}{precautions}"
                        )
                    else:
                        response_text = translate_text(
                            "I couldn't retrieve complete information. Please consult a doctor.",
                            lang
                        )
                else:
                    response_text = translate_text(
                        "No matching conditions found. Please consult a healthcare professional.",
                        lang
                    )
        else:
            response_text = translate_text(
                "Please describe your symptoms so I can help you better.",
                lang
            )
        
        # Format response
        response = {
            "fulfillmentText": response_text,
            "payload": {
                "google": {
                    "expectUserResponse": True,
                    "richResponse": {
                        "items": [{
                            "simpleResponse": {
                                "textToSpeech": response_text,
                                "displayText": response_text
                            }
                        }]
                    }
                }
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}", exc_info=True)
        return jsonify({
            "fulfillmentText": "Sorry, I encountered an error. Please try again."
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)