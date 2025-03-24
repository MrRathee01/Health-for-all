from flask import Flask, request, jsonify
import os
import pandas as pd
from dotenv import load_dotenv

app = Flask(__name__)

PORT = int(os.environ.get('PORT', 8080))

load_dotenv()

# Load datasets
base_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(base_dir, "Dataset1")

# Verify files exist
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH, exist_ok=True)
    print(f"Created directory: {DATASET_PATH}")

try:
    symptom_to_disease = pd.read_csv(os.path.join(DATASET_PATH, "disease.csv"))
except FileNotFoundError:
    print(f"Error: Dataset files not found in {DATASET_PATH}")
    raise
disease_description = pd.read_csv(os.path.join(base_dir, "Dataset1", "symptom_Description.csv"))
precautions = pd.read_csv(os.path.join(base_dir, "Dataset1", "symptom_precaution.csv"))
symptom_severity = pd.read_csv(os.path.join(base_dir, "Dataset1", "Symptom-severity.csv"))

# Group symptoms by disease
grouped_symptoms = symptom_to_disease.groupby("Disease").agg(lambda x: x.dropna().tolist())

def extract_symptoms(text):
    """Extract symptoms from text using the symptom list"""
    symptoms = []
    if not isinstance(text, str):
        return symptoms
    for symptom in symptom_severity["Symptom"]:
        if symptom.lower() in text.lower():
            symptoms.append(symptom)
    return symptoms

def identify_diseases(symptoms):
    """Identify possible diseases based on symptoms"""
    possible_diseases = []
    for disease, row in grouped_symptoms.iterrows():
        disease_symptoms = row[1:]  # Skip disease name column
        if all(symptom in disease_symptoms for symptom in symptoms):
            possible_diseases.append(disease)
    return possible_diseases

def ask_next_symptom(possible_diseases):
    """Get next symptoms to ask to narrow down diagnosis"""
    next_symptoms = set()
    for disease in possible_diseases:
        disease_symptoms = grouped_symptoms.loc[disease][1:]
        next_symptoms.update(disease_symptoms)
    return list(next_symptoms)

def get_disease_info(disease):
    """Get description and precautions for a disease"""
    try:
        description = disease_description[disease_description["Disease"] == disease]["Description"].values[0]
        precaution_columns = [col for col in precautions.columns if col.startswith("Precaution_")]
        disease_precautions = precautions[precautions["Disease"] == disease][precaution_columns].values.flatten()
        precaution_text = ", ".join([p for p in disease_precautions if pd.notna(p)])
        return description, precaution_text
    except:
        return "No description available", "No precautions available"

def check_emergency(symptoms):
    """Check if symptoms indicate an emergency"""
    for symptom in symptoms:
        severity = symptom_severity[symptom_severity["Symptom"] == symptom]["Severity"].values[0]
        if severity == 7:
            return True
    return False

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        req = request.get_json(silent=True, force=True)
        intent_name = req["queryResult"]["intent"]["displayName"]
        session_id = req["session"].split("/")[-1]
        parameters = req["queryResult"]["parameters"]
        
        # Extract symptoms from parameters
        symptoms = []
        for param, value in parameters.items():
            if "symptom" in param.lower():
                if isinstance(value, list):
                    symptoms.extend([s for s in value if s])
                elif isinstance(value, str):
                    symptoms.extend(extract_symptoms(value))
        
        # Remove duplicates
        symptoms = list(set([s for s in symptoms if s]))
        
        if intent_name in ["General Symptoms", "Multiple Symptoms"]:
            if not symptoms:
                response = {"fulfillmentText": "Could you describe your symptoms in more detail?"}
            else:
                possible_diseases = identify_diseases(symptoms)
                if len(possible_diseases) > 1:
                    next_symptoms = ask_next_symptom(possible_diseases)
                    response = {
                        "fulfillmentText": f"Do you also have any of these symptoms: {', '.join(next_symptoms)}?",
                        "outputContexts": [{
                            "name": f"{req['session']}/contexts/awaiting_symptom_confirmation",
                            "lifespanCount": 5,
                            "parameters": {
                                "possible_diseases": possible_diseases,
                                "current_symptoms": symptoms,
                                "next_symptoms": next_symptoms
                            }
                        }]
                    }
                elif len(possible_diseases) == 1:
                    disease = possible_diseases[0]
                    description, precaution_text = get_disease_info(disease)
                    if check_emergency(symptoms):
                        response = {"fulfillmentText": f"EMERGENCY: You might have {disease}. {description}. Precautions: {precaution_text}. Please seek immediate medical attention!"}
                    else:
                        response = {"fulfillmentText": f"You might have {disease}. {description}. Precautions: {precaution_text}."}
                else:
                    response = {"fulfillmentText": "I couldn't identify any matching diseases. Please consult a doctor."}
        
        elif intent_name == "Follow-up Symptoms":
            contexts = req.get("queryResult", {}).get("outputContexts", [])
            symptom_context = next((c for c in contexts if "awaiting_symptom_confirmation" in c["name"]), None)
            
            if symptom_context:
                params = symptom_context.get("parameters", {})
                possible_diseases = params.get("possible_diseases", [])
                current_symptoms = params.get("current_symptoms", [])
                
                # Add new symptoms if provided
                if "followup_symptom" in parameters:
                    new_symptoms = []
                    if isinstance(parameters["followup_symptom"], list):
                        new_symptoms = parameters["followup_symptom"]
                    elif isinstance(parameters["followup_symptom"], str):
                        new_symptoms = extract_symptoms(parameters["followup_symptom"])
                    current_symptoms.extend(new_symptoms)
                
                possible_diseases = identify_diseases(current_symptoms)
                
                if len(possible_diseases) > 1:
                    next_symptoms = ask_next_symptom(possible_diseases)
                    response = {
                        "fulfillmentText": f"Do you also have any of these symptoms: {', '.join(next_symptoms)}?",
                        "outputContexts": [{
                            "name": f"{req['session']}/contexts/awaiting_symptom_confirmation",
                            "lifespanCount": 5,
                            "parameters": {
                                "possible_diseases": possible_diseases,
                                "current_symptoms": current_symptoms,
                                "next_symptoms": next_symptoms
                            }
                        }]
                    }
                elif len(possible_diseases) == 1:
                    disease = possible_diseases[0]
                    description, precaution_text = get_disease_info(disease)
                    if check_emergency(current_symptoms):
                        response = {"fulfillmentText": f"EMERGENCY: You might have {disease}. {description}. Precautions: {precaution_text}. Please seek immediate medical attention!"}
                    else:
                        response = {"fulfillmentText": f"You might have {disease}. {description}. Precautions: {precaution_text}."}
                else:
                    response = {"fulfillmentText": "I couldn't identify any matching diseases. Please consult a doctor."}
            else:
                response = {"fulfillmentText": "Let's start over. What symptoms are you experiencing?"}
        
        elif intent_name == "No More Symptoms":
            contexts = req.get("queryResult", {}).get("outputContexts", [])
            symptom_context = next((c for c in contexts if "awaiting_symptom_confirmation" in c["name"]), None)
            
            if symptom_context:
                params = symptom_context.get("parameters", {})
                possible_diseases = params.get("possible_diseases", [])
                current_symptoms = params.get("current_symptoms", [])
                
                if possible_diseases:
                    disease = possible_diseases[0]
                    description, precaution_text = get_disease_info(disease)
                    if check_emergency(current_symptoms):
                        response = {"fulfillmentText": f"EMERGENCY: You might have {disease}. {description}. Precautions: {precaution_text}. Please seek immediate medical attention!"}
                    else:
                        response = {"fulfillmentText": f"You might have {disease}. {description}. Precautions: {precaution_text}."}
                else:
                    response = {"fulfillmentText": "I couldn't identify any matching diseases. Please consult a doctor."}
            else:
                response = {"fulfillmentText": "Let's start over. What symptoms are you experiencing?"}
        
        else:
            response = {"fulfillmentText": "I didn't understand that. Could you describe your symptoms?"}
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"fulfillmentText": "Sorry, I encountered an error. Please try again."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)