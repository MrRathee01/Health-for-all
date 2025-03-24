from flask import Flask, request, jsonify
import os
import pandas as pd
from dotenv import load_dotenv

app = Flask(__name__)

PORT = int(os.environ.get('PORT', 8080))

load_dotenv()

# Load your datasets (same as in your original script)
base_dir = os.path.dirname(os.path.abspath(__file__))
symptom_to_disease = pd.read_csv(os.path.join(base_dir, "Dataset1", "disease.csv"))
disease_description = pd.read_csv(os.path.join(base_dir, "Dataset1", "symptom_Description.csv"))
precautions = pd.read_csv(os.path.join(base_dir, "Dataset1", "symptom_precaution.csv"))
symptom_severity = pd.read_csv(os.path.join(base_dir, "Dataset1", "Symptom-severity.csv"))
grouped_symptoms = symptom_to_disease.groupby("Disease").agg(lambda x: x.dropna().tolist())

# Helper functions (same as in your original script)
def extract_symptoms(text):
    symptoms = []
    for symptom in symptom_severity["Symptom"]:
        if symptom.lower() in text.lower():
            symptoms.append(symptom)
    return symptoms

def identify_diseases(symptoms):
    possible_diseases = []
    for disease, row in grouped_symptoms.iterrows():
        disease_symptoms = row[1:]
        if all(symptom in disease_symptoms for symptom in symptoms):
            possible_diseases.append(disease)
    return possible_diseases

def ask_next_symptom(possible_diseases):
    next_symptoms = set()
    for disease in possible_diseases:
        disease_symptoms = grouped_symptoms.loc[disease][1:]
        next_symptoms.update(disease_symptoms)
    return list(next_symptoms)

def get_disease_info(disease):
    description = disease_description[disease_description["Disease"] == disease]["Description"].values[0]
    precaution_columns = [col for col in precautions.columns if col.startswith("Precaution_")]
    disease_precautions = precautions[precautions["Disease"] == disease][precaution_columns].values.flatten()
    precaution_text = ", ".join([p for p in disease_precautions if pd.notna(p)])
    return description, precaution_text

def check_emergency(symptoms):
    for symptom in symptoms:
        severity = symptom_severity[symptom_severity["Symptom"] == symptom]["Severity"].values[0]
        if severity == 7:
            return True
    return False

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    intent_name = req["queryResult"]["intent"]["displayName"]
    session_id = req["session"].split("/")[-1]
    user_input = req["queryResult"]["queryText"]
    
    # Extract symptoms from user input
    symptoms = extract_symptoms(user_input)
    
    if intent_name == "Symptoms Description":
        if not symptoms:
            response = "Could you describe your symptoms in more detail?"
        else:
            possible_diseases = identify_diseases(symptoms)
            if len(possible_diseases) > 1:
                next_symptoms = ask_next_symptom(possible_diseases)
                response = {
                    "followupEventInput": {
                        "name": "ask_followup",
                        "parameters": {
                            "possible_diseases": possible_diseases,
                            "next_symptoms": next_symptoms
                        }
                    }
                }
            elif len(possible_diseases) == 1:
                disease = possible_diseases[0]
                description, precaution_text = get_disease_info(disease)
                if check_emergency(symptoms):
                    response = f"You might have {disease}. {description}. Precautions: {precaution_text}. This seems serious. Please seek emergency help!"
                else:
                    response = f"You might have {disease}. {description}. Precautions: {precaution_text}."
            else:
                response = "I couldn't identify any matching diseases. Please consult a doctor."
    
    elif intent_name == "Follow-up Symptoms":
        # Get context from previous intent
        parameters = req["queryResult"]["outputContexts"][0]["parameters"]
        possible_diseases = parameters["possible_diseases"]
        next_symptoms = parameters["next_symptoms"]
        
        if "yes" in user_input.lower():
            # User confirmed additional symptoms
            new_symptoms = extract_symptoms(user_input)
            symptoms.extend(new_symptoms)
            possible_diseases = identify_diseases(symptoms)
            
            if len(possible_diseases) > 1:
                next_symptoms = ask_next_symptom(possible_diseases)
                response = {
                    "followupEventInput": {
                        "name": "ask_followup",
                        "parameters": {
                            "possible_diseases": possible_diseases,
                            "next_symptoms": next_symptoms
                        }
                    }
                }
            elif len(possible_diseases) == 1:
                disease = possible_diseases[0]
                description, precaution_text = get_disease_info(disease)
                if check_emergency(symptoms):
                    response = f"You might have {disease}. {description}. Precautions: {precaution_text}. This seems serious. Please seek emergency help!"
                else:
                    response = f"You might have {disease}. {description}. Precautions: {precaution_text}."
            else:
                response = "I couldn't identify any matching diseases. Please consult a doctor."
        else:
            # User doesn't have additional symptoms
            if len(possible_diseases) > 0:
                disease = possible_diseases[0]  # Take the first match
                description, precaution_text = get_disease_info(disease)
                if check_emergency(symptoms):
                    response = f"You might have {disease}. {description}. Precautions: {precaution_text}. This seems serious. Please seek emergency help!"
                else:
                    response = f"You might have {disease}. {description}. Precautions: {precaution_text}."
            else:
                response = "I couldn't identify any matching diseases. Please consult a doctor."
    
    else:
        response = "I didn't understand that. Could you describe your symptoms?"
    
    return jsonify({"fulfillmentText": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)