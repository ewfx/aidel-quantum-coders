from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import requests
import spacy
import numpy as np
import json
import time
from sklearn.ensemble import IsolationForest
from transformers import pipeline
import re
import requests
import json
import pandas as pd
from fuzzywuzzy import fuzz
import uuid

#from sec_edgar import Edgar


# Initialize FastAPI app
app = FastAPI()

# Load spaCy and Hugging Face models
nlp_spacy = spacy.load("en_core_web_sm")
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# API Endpoints
OPEN_CORPORATES_API = "https://api.opencorporates.com/v0.4/companies/search"
SEC_EDGAR_SEARCH_API = "https://www.sec.gov/cgi-bin/browse-edgar?company="
OFAC_SANCTIONS_LIST = "https://www.treasury.gov/ofac/downloads/sdnlist.txt"
SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"

# Cache API responses to reduce redundant calls
api_cache = {}

def fetch_with_cache(url):
    """ Fetch API data with caching to avoid redundant requests. """
    if url in api_cache:
        return api_cache[url]
    
    response = requests.get(url)
    if response.status_code != 200:
        return None  # Return None if API call fails
    
    api_cache[url] = response.text
    time.sleep(1)  # Avoid rate limits
    return response.text



def load_ofac_entities(csv_path):
    df = pd.read_csv("ofac_sanctions_list.csv")

    # Clean data (removing unwanted placeholders)
    df = df.applymap(lambda x: str(x).replace('-0-', '').strip())

    # Extract only relevant columns (assuming first column is ID, second is entity name)
    df = df.iloc[:, :2]
    df.columns = ["Entity_ID", "Entity_Name"]
    #print(df.head())
    return set(df["Entity_Name"].str.upper().dropna().unique())  # Store as uppercase for case-insensitive matching

# Load data once

#-----------------------

def extract_entities_1(data, ofac_entities):
    """ Extract valid entities while ensuring full company names are taken into account. """
    text = " .LTD   ".join(str(value) for value in data.values()) if isinstance(data, dict) else str(data)
    print(text);
    # Use spaCy NER
    doc_spacy = nlp_spacy(text)
    entities_orgs = {ent.text.strip() for ent in doc_spacy.ents if ent.label_ == "ORG"}
    entities_persons = {ent.text.strip() for ent in doc_spacy.ents if ent.label_ == "PERSON"}
    print(entities_orgs)
    # Use Hugging Face NER
    entities_hf = []
    current_entity = []
    previous_tag = None
    entity_type = None  

    for ent in ner_pipeline(text):
        word = ent["word"].replace("##", "")  
        word = re.sub(r"[^a-zA-Z0-9&.,-]", "", word)  
        entity_tag = ent["entity"]

        if entity_tag.startswith("B-"):
            if current_entity:
                entities_hf.append((" ".join(current_entity).strip(), entity_type))  
            current_entity = [word]  
            entity_type = entity_tag[2:]  
        elif entity_tag.startswith("I-") and entity_type == entity_tag[2:]:  
            current_entity.append(word)
        else:
            # If the entity type changes unexpectedly, finalize the previous entity
            if current_entity:
                entities_hf.append((" ".join(current_entity).strip(), entity_type))
            current_entity = []
            entity_type = None  # Reset entity tracking


        previous_tag = entity_tag

    if current_entity:
        entities_hf.append((" ".join(current_entity).strip(), entity_type))  

    # Separate ORGs and PERSONs from Hugging Face results
    entities_hf_orgs = {e[0] for e in entities_hf if e[1] == "ORG"}
    entities_hf_persons = {e[0] for e in entities_hf if e[1] == "PERSON"}

    print(entities_hf_orgs)
    print("spacy:");
    print( entities_orgs);
    #print(entities_hf_orgs);
    # Combine results from both models
    extracted_orgs = list(entities_orgs.union(entities_hf_orgs))
    extracted_persons = list(entities_persons.union(entities_hf_persons))

    # Remove common location-based words
    unwanted_words = {"mountain", "view", "ca", "amphitheatre", "parkway", "redmond", "wa", "santa", "clara"}
    extracted_orgs = [
        " ".join(word for word in e.split() if word.lower() not in unwanted_words)
        for e in extracted_orgs
    ]

    #Ensure full company names are retained and allow single-word matches if in OFAC
    extracted_orgs = [
        e.strip() for e in extracted_orgs 
        if len(e.split()) > 1 or e.upper() in ofac_entities  
    ]

    if not extracted_orgs and not extracted_persons:
        raise HTTPException(status_code=422, detail="No valid entities found in the provided data.")
    print(extracted_orgs)
    print(extracted_persons)
    return extracted_orgs 

#---------------------------
def extract_entities(data, ofac_entities):
    """ Extract valid entities while ensuring full company names are considered. """
    text = " .LTD   ".join(str(value) for value in data.values()) if isinstance(data, dict) else str(data)

    # Use spaCy NER
    doc_spacy = nlp_spacy(text)
    entities_orgs = {ent.text.strip() for ent in doc_spacy.ents if ent.label_ in {"ORG", "GPE"}}
    entities_persons = {ent.text.strip() for ent in doc_spacy.ents if ent.label_ == "PERSON"}

    print(entities_orgs)
    # Use Hugging Face NER
    entities_hf = []
    current_entity = []
    entity_type = None  

    for ent in ner_pipeline(text):
        word = ent["word"].replace("##", "")
        word = re.sub(r"[^a-zA-Z0-9&.,-]", "", word)  
        entity_tag = ent["entity"]

        if entity_tag.startswith("B-"):
            if current_entity:
                entities_hf.append((" ".join(current_entity).strip(), entity_type))
            current_entity = [word]
            entity_type = entity_tag[2:]
        elif entity_tag.startswith("I-") and entity_type == entity_tag[2:]:
            current_entity.append(word)
        else:
            if current_entity:
                entities_hf.append((" ".join(current_entity).strip(), entity_type))
            current_entity = []
            entity_type = None  

    if current_entity:
        entities_hf.append((" ".join(current_entity).strip(), entity_type))

    # Separate ORGs and PERSONs from Hugging Face results
    entities_hf_orgs = {e[0] for e in entities_hf if e[1] in {"ORG", "GPE"}}
    entities_hf_persons = {e[0] for e in entities_hf if e[1] == "PERSON"}


    print(entities_hf_orgs)
    # Merge results from both models
    extracted_orgs = list(entities_orgs.union(entities_hf_orgs))
    extracted_persons = list(entities_persons.union(entities_hf_persons))

    # Identify embassies separately
    extracted_orgs = [
        e.strip() for e in extracted_orgs
        if len(e.split()) > 1 or e.upper() in ofac_entities or is_government_agency(e)
    ]

    if not extracted_orgs and not extracted_persons:
        raise ValueError("No valid entities found in the provided data.")
    
    return extracted_orgs



def enrich_entity(entity_name):
    """ Fetch company details from OpenCorporates API. """
    response = requests.get(OPEN_CORPORATES_API, params={"q": entity_name})
    if response.status_code == 200:
        data = response.json()
        return data.get("results", {}).get("companies", [])[:1]  # Avoid KeyErrors
    return None

def check_ofac_sanctions(entity_name):
    # """ Check if an entity is listed in the OFAC sanctions list. """
    # sanctions_list = fetch_with_cache(OFAC_SANCTIONS_LIST)
    # if not sanctions_list:
    #     return False
    # return any(entity_name.lower() in line.lower() for line in sanctions_list.split("\n"))
    df = pd.read_csv("ofac_sanctions_list.csv")

    # Clean data (removing unwanted placeholders)
    df = df.applymap(lambda x: str(x).replace('-0-', '').strip())

    # Extract only relevant columns (assuming first column is ID, second is entity name)
    df = df.iloc[:, :2]
    df.columns = ["Entity_ID", "Entity_Name"]
    #print(df.head())
    df["Entity_Name"] = df["Entity_Name"].str.strip().str.lower()

    # Function to search for an entity
    entity_name = entity_name.strip().lower()
    
    # Exact match search
    if entity_name in df["Entity_Name"].values:
        return True  # Exact match found
    
    # Partial match search (checking substrings)
    fuzzy_matches = df[df["Entity_Name"].apply(lambda x: fuzz.partial_ratio(entity_name, x) >= 90)]

    if not fuzzy_matches.empty:
        print("Possible fuzzy matches:", fuzzy_matches["Entity_Name"].tolist())
        return True  # Found close match

    return False  # No match found


def is_government_agency(entity_name):
    """Check if an entity is classified as a government agency in Wikidata"""
    query = f'''
    SELECT ?item WHERE {{
        ?item ?label "{entity_name}"@en.
        ?item wdt:P31 wd:Q327333.  # "instance of" -> "government agency"
    }}
    '''
    url = "https://query.wikidata.org/sparql"
    headers = {"User-Agent": "EntityClassifier/1.0"}
    response = requests.get(url, params={"query": query, "format": "json"}, headers=headers)
    
    if response.status_code == 200 and response.json()["results"]["bindings"]:
        print("It is government agency")
        return True  # Found in Wikidata as a government agency

    return False


irs_file_path = "irs_data.txt"  # Path to your uploaded file


def search_non_profit(entity_name):
    """Check if an entity exists in the IRS non-profit database."""
    df = pd.read_csv(irs_file_path, sep="|", header=None, dtype=str)

    # Assign column names based on the structure of the file
    df.columns = ["EIN", "Organization Name", "City", "State", "Country", "Category"]
    fuzzy_matches = df[df["Organization Name"].apply(lambda x: fuzz.partial_ratio(entity_name, x) >= 90)]

    if not fuzzy_matches.empty:
        print("Possible fuzzy matches:", fuzzy_matches["Entity_Name"].tolist())
        return True  # Found close match

    return False  # No match found

    

def get_sec_cik(company_name):
    
    """Fetch CIK for a given company name from SEC ticker data."""
    
    headers = {"User-Agent": "YourName (your_email@example.com)"}
    response = requests.get(SEC_TICKER_URL, headers=headers)

    if response.status_code == 200:
        data = response.json()
        for entry in data.values():
            if company_name.lower() in entry["title"].lower():
                cik = str(entry["cik_str"]).zfill(10)  # Ensure 10-digit CIK
                print(cik)
                return cik

        return {"Not found"}
    else:
        return {"error": f"Failed to fetch SEC data: {response.status_code}"}





def check_sec_edgar_filings(cik):
    """ Check if a company has SEC filings using CIK. """
    if not cik:
        return False

    # url = SEC_COMPANY_FACTS_API.format(cik=cik)
    # response = requests.get(url, headers=HEADERS)

    # if response.status_code == 200:
    #     data = response.json()
    #     return bool(data)  # If data exists, the company has SEC filings.
    # return False



def detect_anomalies(features):
    """ Detect anomalous transactions using Isolation Forest. """
    model = IsolationForest(contamination=0.05, random_state=42)
    return model.fit_predict(features)

def assign_risk_score(entity_name, enriched_data, ofac_flag, cik):
    """ Calculate risk score based on entity details and external data. """
    if isinstance(cik, int) or (isinstance(cik, str) and cik.isdigit()):
        cik_present = True  # CIK is a valid number
    else:
        cik_present = False  # Invalid CIK (not found or non-numeric)

    score = 20  # Base risk score
    
    if ofac_flag:
        score += 40
    if not cik_present:
        score += 15
    if enriched_data:
        company_details = enriched_data[0].get("company", {})
        if company_details.get("status") in ["inactive", "dissolved"]:
            score += 20  
        if not company_details.get("officers"):
            score += 10  
    return min(score, 100)
    
    
    # if cik and not ofac_flag:
    #     return 10  # Low risk for SEC-listed, non-sanctioned companies
    # elif cik and ofac_flag:
    #     return 50  # Medium risk if SEC-listed but found in OFAC
    # elif not cik and not ofac_flag:
    #     return 65  # Higher risk if not in SEC but not in OFAC
    # else:
    #     return 85  # Very high risk if company is in OFAC sanctions list


def generate_evidence(entity_name, enriched_data, ofac_flag, cik_present, gov, non_profit, evidence, supportingEvidence):
    """ Provide evidence supporting risk classification. """
   
    if enriched_data:
        company = enriched_data[0].get("company", {})
        #evidence.append(f"Company {entity_name} found in OpenCorporates with status: {company.get('status')}.")
    # if ofac_flag:
    #     evidence.append(f"Entity {entity_name} is on the OFAC sanctions list.")
    # if cik_present :
    #     evidence.append(f"No SEC filings found for {entity_name}, raising compliance concerns.")
    # if anomaly_detected:
    #     evidence.append(f"Entity {entity_name} flagged as anomalous based on transaction patterns.")
    
    if ofac_flag:
        evidence.append("OFAC Sanctions List");
        supportingEvidence.append(f"Entity {entity_name} is on the OFAC sanctions list.")
    elif (cik_present):
        evidence.append("SEC Edgar Filings")
        #supportingEvidence.append(f"SEC filings found for {entity_name}.")
    elif (not cik_present):
        supportingEvidence.append(f"SEC filings not found for {entity_name}.")
    elif(gov):
        evidence.append("WikiData for Govenement organisation")
        supportingEvidence.append(f"{entity_name} is a Govenement organisation according to Wikidata.")
    elif(non_profit) :
        evidence.append("IRS Non Profit Oraganisations data")
        supportingEvidence.append(f"{entity_name} is a non profit oragnisation ccording to IRS data.")

class EntityRequest(BaseModel):
    data: Optional[Dict[str, Any]] = None  # Ensure `data` is optional but must be a dictionary if present

@app.post("/analyze-entities")
async def process_entities(request: EntityRequest):
    """ API Endpoint: Extract entities, assess risk, return JSON response. """
    if not request.data:
        raise HTTPException(status_code=400, detail="Invalid request: 'data' is required and must be a JSON object.")

    data = request.data
    print("Received Data:", data)  # Debugging log
    ofac_entities = load_ofac_entities("ofac_sanctions_list.csv")
    entities = extract_entities(data, ofac_entities)
    print("Entities:", entities)
    results = []

    if not entities:
        raise HTTPException(status_code=400, detail="No valid entities found in the provided data.")

    # Generate mock feature vectors for anomaly detection
    features = np.random.rand(len(entities), 5)  # Replace with actual risk features
    anomaly_predictions = detect_anomalies(features)

    # return {
    #     "transactionID": transaction_id,
    #     "extractedEntity": data.entity,
    #     "entityType": data.entityType,
    #     "riskScore": final_risk_score,
    #     "confidenceScore": final_confidence,
    #     "reason": finbert_reason,
    #     "supportingEvidence": {
    #         "SEC Edgar": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={data.entity}",
    #         "OFAC Sanctions": "https://sanctionssearch.ofac.treas.gov/"
    #     } 
    # }

    transaction_id = str(uuid.uuid4())
    extractedEntity = entities
    entity_type = []
    riskScores = []
    evidences = []
    supportingEvidence = []

    for i,entity in enumerate(entities):
        print("entity:", entity)
        print("ENTITY")
        print(entities)
        print("ENTITIES")
        enriched_data = enrich_entity(entity)
        cik = get_sec_cik(entity)
        sec_filings = check_sec_edgar_filings(cik) if cik else False
        ofac_flag = check_ofac_sanctions(entity)
        gov = is_government_agency(entity)
        non_profit = search_non_profit(entity)
        print(gov, non_profit)
        if isinstance(cik, int) or (isinstance(cik, str) and cik.isdigit()):
            cik_present = True  # CIK is a valid number
        else:
            cik_present = False 
            
        if (cik_present):
             entity_type.append("Corporation")
        elif (ofac_flag) :
            entity_type.append("Shell Company")
        elif(gov):
            entity_type.append("Government Agency")
        elif(non_profit) :
            entity_type.append("Non profit organisation")
        #is_anomalous = bool(anomaly_predictions[i] == -1)
        
        risk_score = assign_risk_score(entity, enriched_data, ofac_flag, cik)
        riskScores.append(risk_score)
        #evidence = generate_evidence(entity, enriched_data, ofac_flag, sec_filings, is_anomalous)
        generate_evidence(entity, enriched_data, ofac_flag, cik_present, gov, non_profit, evidences, supportingEvidence)

        # results.append({
        #     "entity": entity,
        #     "risk_score": risk_score,
        #     "confidence_score": round(1 - (risk_score / 100), 2),
        #     "is_anomalous": is_anomalous,
        #     "evidence": evidence,
        #     "supportingEvidence": {
        #         "SEC Edgar": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}" if cik else "Not Found",
        #         "OFAC Sanctions": "https://sanctionssearch.ofac.treas.gov/" if ofac_flag else "Not Listed"
        #     }
    results.append({
        "Transaction ID": transaction_id,
        "Extracted Entity": extractedEntity,
        "Entity Type": entity_type,
        "Risk Score": riskScores,
        "Supporting Evidence": evidences,
        "Confidence Score": round(1 - (risk_score / 100), 2),
        "Reason": supportingEvidence
    })

    return results
