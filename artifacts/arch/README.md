 📌 **Detailed Explanation of the AI Driven Entity Intelligence Risk Analysis**
 
The system processes a financial transaction text, extracts entities, classifies them, assigns a risk score, and provides supporting evidence. 

Below is a step-by-step breakdown of the flow:

📌 **Summary of the Flow**

✅ Input: Raw transaction text
 ✅ Step 1: Extract named entities (spaCy + Hugging Face)
 ✅ Step 2: Classify entities using IRS, SEC, OFAC, Wikidata
 ✅ Step 3: Assign a risk score
 ✅ Step 4: Generate supporting evidence
 ✅ Output: JSON response with entity classifications, risk scores & evidence

1️⃣ API Receives Transaction Text
The FastAPI service receives a request containing unstructured transaction text (e.g., "Embassy of Argentina in Uruguay pays 500 million to Apple Inc.").


This text is passed as input to the entity extraction module.



2️⃣ **Extract Entities**
Goal: Identify relevant entities (companies, organizations, embassies, etc.).
Uses spaCy for Named Entity Recognition (NER).


Uses Hugging Face Transformers (BERT-based model) for improved entity detection.


Extracted entities can be:


Organizations (ORG) → e.g., Apple Inc.


Geopolitical Entities (GPE) → e.g., Argentina, Uruguay


Persons (PERSON) → e.g., Individual names (if mentioned)


🔹 **Example Output:**
{
  "ExtractedEntities": ["Department of Disaster Management" "Apple Inc."]
}


3️⃣ **Classify Entities**
Each extracted entity is classified into one of four categories:
📌 Corporation (Public/Private Company)
✅ SEC CIK Lookup:


If an entity is listed in the SEC Ticker database, it is classified as a Corporation.


Calls get_sec_cik(entity_name) to fetch the Central Index Key (CIK).


If a valid CIK is found → Corporation


If no CIK found → Move to other checks.


📌 Non-Profit Organization
✅ IRS Non-Profit Check:


Calls search_non_profit(entity_name) to check the IRS database.


If found → Entity is classified as a Non-Profit.


📌 Government Agency
✅ Wikidata Query:


Calls is_government_agency(entity_name) using SPARQL query on Wikidata.


If found → Classified as a Government Agency.


📌 Shell Company (High Risk)


✅ OFAC Sanctions List Check:


Calls check_ofac_sanctions(entity_name) to verify if the entity is on the OFAC watchlist.


If found → Classified as a Shell Company (High Risk).


🔹 Example Output:
{
  "EntityTypes": {
    "Apple Inc.": "Corporation",
    "Embassy of Argentina in Uruguay": "Government Agency"
  }
}


4️⃣ **Assign Risk Score**
Goal: Calculate a risk score (0-100) for each entity.
Risk scoring logic:
+40 points → If entity is in the OFAC Sanctions List.


+15 points → If the entity lacks a CIK (not SEC-listed).


+20 points → If the company status is inactive or dissolved (via OpenCorporates).


+10 points → If no officers/founders found in external databases.


10 points (LOWEST RISK) → If entity is a Non-Profit.


Max Score: 100 (Higher score = Greater risk).


🔹 Example Output:
{
  "RiskScores": {
    "Apple Inc.": 20,
    "Embassy of Argentina in Uruguay": 10
  }
}


5️⃣ **Generate Supporting Evidence**
For each classified entity, the system generates relevant supporting evidence:
SEC Edgar Filings → If the company has a valid CIK.


OFAC Sanctions List → If the entity is sanctioned.


IRS Non-Profit Database → If the entity is a verified Non-Profit.


Wikidata → If the entity is a government agency.


🔹 Example Output:
{
  "SupportingEvidence": {
    "Apple Inc.": ["SEC Edgar Filings"],
    "Embassy of Argentina in Uruguay": ["Wikidata for Government Organizations"]
  }
}


6️⃣ **API Returns Final Response**
The system compiles all the extracted information and returns it in a structured JSON format:
{
  "Transaction ID": "3fbdca21-2c4d-41d7-9c91-30823d2a1b19",
  "Extracted Entities": ["Embassy of Argentina in Uruguay", "Apple Inc."],
  "Entity Types": {
    "Apple Inc.": "Corporation",
    "Embassy of Argentina in Uruguay": "Government Agency"
  },
  "Risk Score": 20
  "Supporting Evidence": {
    "Apple Inc.": ["SEC Edgar Filings"],
    "Embassy of Argentina in Uruguay": ["Wikidata for Government Organizations"]
  },
  "Confidence Scores: 0.8,
  "Reasons": {
    "SEC filings found for Apple Inc and Embassy of Argentina in Uruguay identified as a government entity in Wikidata"
  }
}




