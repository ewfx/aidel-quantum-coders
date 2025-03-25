import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch

# Force CPU usage (fixes MPS issues on Mac)
device = "cpu"

# Load GPT-2 Model for text generation
gpt2_pipeline = pipeline("text-generation", model="distilgpt2", device=torch.device(device))


app = FastAPI()

# Function to generate risk score using GPT-2
def risk_assessment(entity, transaction_details):
    """Generates a risk score using GPT-2"""
    prompt = f"Risk assessment for {entity} based on {transaction_details}:"
    response = gpt2_pipeline(prompt, max_length=100)[0]["generated_text"]

    # Simple risk logic based on response content
    if any(word in response.lower() for word in ["sanctions", "fraud", "russia", "money laundering", "terrorist financing"]):
        risk_score = "High Risk"
        confidence_score = 0.60
        reason = "Potential fraud or sanctioned entity involvement."
    elif any(word in response.lower() for word in ["large transaction", "offshore", "cayman", "suspicious", "shell company"]):
        risk_score = "Medium Risk"
        confidence_score = 0.78
        reason = "Frequent high-value transactions detected."
    else:
        risk_score = "Low Risk"
        confidence_score = 0.95
        reason = "No significant risk detected."

    return risk_score, confidence_score, reason

class RiskData(BaseModel):
    entity: str
    entityType: str
    transactionDetails: str

@app.post("/risk_score")
def get_risk_score(data: RiskData):
    transaction_id = str(uuid.uuid4())

    # Perform risk scoring
    risk_score, confidence_score, reason = risk_assessment(data.entity, data.transactionDetails)

    # Supporting evidence
    supporting_evidence = {
        "OpenCorporates": f"https://opencorporates.com/companies/search?q={data.entity}",
        "SEC Edgar": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={data.entity}",
        "OFAC Sanctions": "https://sanctionssearch.ofac.treas.gov/"
    }

    return {
        "transactionID": transaction_id,
        "extractedEntity": data.entity,
        "entityType": data.entityType,
        "riskScore": risk_score,
        "confidenceScore": confidence_score,
        "reason": reason,
        "supportingEvidence": supporting_evidence
    }

# Run this file with: uvicorn risk_scoring:app --reload
