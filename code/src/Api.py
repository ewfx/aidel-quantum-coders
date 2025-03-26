@app.get("/process-emails")
def process_emails():
    """Fetch and classify emails"""
    emails = fetch_emails()
    results = []

    for email in emails:
        content = email["body"]
        
        # Process attachments
        for filename, file_content in email["attachments"]:
            content += "\n" + process_attachment(filename, file_content)
        
        # Classify email/document
        classification = classify_text(content)
        routed_to = route_document(classification, email["sender"])
        
        result = {
            "subject": email["subject"],
            "sender": email["sender"],
            "classification": classification,
            "routed_to": routed_to,
        }
        results.append(result)
    
    return results
