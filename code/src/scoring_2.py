from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from transformers import pipeline
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder

# Initialize FastAPI app
app = FastAPI()

device = "cpu"

# Load FinBERT once at startup
finbert_pipeline = pipeline("text-classification", model="ProsusAI/finbert")

class RiskData(BaseModel):
    entity: str
    entityType: str
    transactionAmount: float
    transactionCountry: str
    transactionType: str
    transactionDetails: str

# Fetch SEC Edgar data

def fetch_sec_data(company_name):
    sec_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={company_name}"
    headers = {"User-Agent": "MyCompanyBot/1.0"}
    response = requests.get(sec_url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        filings = soup.find_all("tr")[1:]  # Ignore table header
        
        if filings:
            last_filing_date = filings[0].find_all("td")[3].text.strip()
            return {"filings": len(filings), "last_filing_date": last_filing_date}
    return {"filings": 0, "last_filing_date": "Unknown"}

# OFAC sanctions check (using local SDN list parsing)
def check_ofac_sanctions(entity_name):
    sanctions_list_url = "https://www.treasury.gov/ofac/downloads/sdnlist.txt"
    response = requests.get(sanctions_list_url)
    
    if response.status_code == 200:
        sdn_data = response.text
        return {"sanctioned": entity_name.lower() in sdn_data.lower()}
    
    return {"sanctioned": False}

# Train XGBoost model
def train_xgboost():
    data = {
        "transactionAmount": [5000, 100000, 200000, 750000, 15000, 250000],
        "transactionCountry": ["USA", "Cayman Islands", "Russia", "UK", "India", "Switzerland"],
        "transactionType": ["Wire Transfer", "Crypto", "Offshore", "Card Payment", "Bank Transfer", "Wire Transfer"],
        "riskScore": [0, 1, 1, 0, 0, 1]  # 1 = High Risk, 0 = Low Risk
    }
    df = pd.DataFrame(data)
    
    le_country = LabelEncoder()
    le_type = LabelEncoder()
    df["transactionCountry"] = le_country.fit_transform(df["transactionCountry"])
    df["transactionType"] = le_type.fit_transform(df["transactionType"])
    
    X = df[["transactionAmount", "transactionCountry", "transactionType"]]
    y = df["riskScore"]
    model = xgb.XGBClassifier()
    model.fit(X, y)
    
    return model, le_country, le_type

xgb_model, country_encoder, type_encoder = train_xgboost()

# FinBERT Risk Assessment
def finbert_risk_assessment(entity, transaction_details):
    result = finbert_pipeline(f"{entity} - {transaction_details}")
    sentiment = result[0]['label']
    confidence_score = result[0]['score']
    risk_score = "High Risk" if sentiment == "Negative" else "Low Risk"
    reason = "Potential fraud detected." if sentiment == "Negative" else "No significant risk."
    print(f"FinBERT Sentiment: {sentiment}, Confidence: {confidence_score}")

    return risk_score, confidence_score, reason

def xgboost_risk_assessment(amount, country, transaction_type):
    # Encode categorical features
    country_encoded = country_encoder.transform([country])[0] if country in country_encoder.classes_ else 0
    type_encoded = type_encoder.transform([transaction_type])[0] if transaction_type in type_encoder.classes_ else 0

    # Prepare input for XGBoost
    input_data = np.array([[amount, country_encoded, type_encoded]])
    
    # Get risk probability
    risk_prediction = xgb_model.predict_proba(input_data)[0][1]  # Probability of High Risk

    print(f"XGBoost Input: {input_data}, Risk Prediction: {risk_prediction}")  # Debugging

    return risk_prediction

@app.post("/risk_score")
def get_risk_score(data: RiskData):
    transaction_id = str(uuid.uuid4())
    
    sec_data = fetch_sec_data(data.entity)
    ofac_data = check_ofac_sanctions(data.entity)
    finbert_risk, finbert_confidence, finbert_reason = finbert_risk_assessment(data.entity, data.transactionDetails)
    # Run XGBoost risk assessment
    xgb_risk_score = xgboost_risk_assessment(data.transactionAmount, data.transactionCountry, data.transactionType)

    # Determine final risk score based on multiple factors
    if ofac_data["sanctioned"]:
        final_risk_score = "High Risk"
    elif xgb_risk_score > 0.75:  # More aggressive threshold for high-risk transactions
        final_risk_score = "High Risk"
    elif xgb_risk_score > 0.4 and finbert_risk == "High Risk":  # If both XGBoost & FinBERT indicate concerns
        final_risk_score = "Medium Risk"
    elif xgb_risk_score > 0.3 or finbert_risk == "Neutral":  # Medium Risk if either gives slight concern
        final_risk_score = "Medium Risk"
    else:
        final_risk_score = "Low Risk"

    #final_risk_score = "High Risk" if ofac_data["sanctioned"] else finbert_risk
    final_confidence = round(finbert_confidence, 2)
    
    return {
        "transactionID": transaction_id,
        "extractedEntity": data.entity,
        "entityType": data.entityType,
        "riskScore": final_risk_score,
        "confidenceScore": final_confidence,
        "reason": finbert_reason,
        "supportingEvidence": {
            "SEC Edgar": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={data.entity}",
            "OFAC Sanctions": "https://sanctionssearch.ofac.treas.gov/"
        }
    }
