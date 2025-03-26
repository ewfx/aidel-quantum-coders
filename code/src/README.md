## Steps to setup and run

Step 1: python3 -m pip install fastapi uvicorn transformers xgboost pandas numpy requests beautifulsoup4 scikit-learn

Step 2: I faced an error regarding absence of libomp, if you face the error brew install libomp

Step 3: cd code/src 
uvicorn risk_scoring:app --reload  

Step 4: 
Sample data format for now in postman
http://127.0.0.1:8000/risk_score

 {
    "entity": "ABC Corp",
    "entityType": "Company",
    "transactionAmount": 500000,
    "transactionCountry": "Russia",
    "transactionType": "Crypto",
    "transactionDetails": "Large crypto transfer to offshore account"
}

