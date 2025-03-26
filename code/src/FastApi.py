import os
import json
import requests
from fastapi import FastAPI, HTTPException
from imapclient import IMAPClient
from email import message_from_bytes
from dotenv import load_dotenv
from tika import parser
import fitz  # PyMuPDF
from langchain.llms import LlamaCpp

# Load environment variables
load_dotenv()

# Llama API Key
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")

# FastAPI app
app = FastAPI()

# Initialize Llama LLM
llm = LlamaCpp(model_path="path/to/your/llama/model.bin")  # Local or API-based

# Email Config
EMAIL_SERVER = os.getenv("IMAP_SERVER")
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
