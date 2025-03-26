def classify_text(text):
    """Use Llama AI to classify email/document type"""
    prompt = f"""
    Analyze the following email/document content and classify it into categories:
    - Invoice
    - Support Request
    - Contract
    - HR-related
    - Spam

    Content:
    {text}

    Respond with the category name.
    """

    response = llm(prompt)
    return response.strip()
