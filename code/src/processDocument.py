def process_attachment(filename, content):
    """Extract text from PDF or Word documents"""
    text = ""

    # Process PDF
    if filename.endswith(".pdf"):
        pdf_document = fitz.open(stream=content, filetype="pdf")
        for page in pdf_document:
            text += page.get_text()

    # Process Word (.docx)
    elif filename.endswith(".docx"):
        parsed = parser.from_buffer(content)
        text = parsed["content"]

    return text
