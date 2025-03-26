# 🚀 AI Driven Entity Intelligence Risk Analysis

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
This project targets AI Driven Entity Risk Analysis. 

The system processes financial transaction text, extracts entities, classifies them, assigns a risk score, and provides supporting evidence.

The purpose is to determine high risk transactions and flag then by publishing a risk score.

## 🎥 Demo
🔗 [Live Demo](#) (if applicable)  
📹 [Video Demo](#) (if applicable)  
🖼️ Screenshots:

![Screenshot 1](link-to-image)

Screenshots attached in artifacts/demo

## 💡 Inspiration
What inspired you to create this project? Describe the problem you're solving.

## ⚙️ What It Does
Summary of the Flow

✅ Input: Raw transaction text

✅ Step 1: Extract named entities (spaCy + Hugging Face)

✅ Step 2: Classify entities using IRS, SEC, OFAC, Wikidata

✅ Step 3: Assign a risk score

✅ Step 4: Generate supporting evidence

✅ Output: JSON response with entity classifications, risk scores & evidence.

## 🛠️ How We Built It
Technology: Python

Model used for entity extraction: spaCY + Hugging face

LocalAPI testing: Postman

## 🚧 Challenges We Faced

1. Obtaining the data in order to classify the entities to right entity type. Most of the data didn't have workable APIs or documentation on how to use them. Had to download it and parse the CSV
2. Extraction large multi word organisations as they used to be extracted as individual words and not recognised
3. Grouping of multiple company names that we close in the input data



## 🏃 How to Run

1. Install dependencies  
   ```sh
   pip install fastapi uvicorn requests pandas numpy scikit-learn transformers fuzzywuzzy spacy
   python -m spacy download en_core_web_sm

   ```
2. Run the project  
   ```sh
   cd to src
   uvicorn risk_scoring:app --reload
   ```

## 🏗️ Tech Stack
- 🔹 Backend: Python
- 🔹 Models: spaCY, Hugging Face
- 🔹 Other: FastAPI, Postman

## 👥 Team
- **Subhransu Panda** - [GitHub](#) 
- **Hanumanthiah P** - [GitHub](#) 
- **Chethana** - [GitHub](#) 
- **Purushotham** - [GitHub](#) 
- **Shashank M** - [GitHub](#) 
