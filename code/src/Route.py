def route_document(classification, sender):
    """Route emails/documents based on classification"""
    routes = {
        "Invoice": "finance@company.com",
        "Support Request": "support@company.com",
        "Contract": "legal@company.com",
        "HR-related": "hr@company.com",
        "Spam": "spam@company.com",
    }
    
    routed_to = routes.get(classification, "unknown@company.com")
    return routed_to
