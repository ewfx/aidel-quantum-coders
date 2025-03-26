def fetch_emails():
    """Fetch unread emails and extract content"""
    emails = []
    with IMAPClient(EMAIL_SERVER) as client:
        client.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        client.select_folder("INBOX", readonly=False)
        messages = client.search(["UNSEEN"])  # Fetch unread emails
        
        for msgid, data in client.fetch(messages, ["RFC822"]).items():
            email_msg = message_from_bytes(data[b"RFC822"])
            subject = email_msg["subject"]
            sender = email_msg["from"]
            body = None

            # Extract body (plaintext or HTML)
            if email_msg.is_multipart():
                for part in email_msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode()
            else:
                body = email_msg.get_payload(decode=True).decode()

            # Extract attachments
            attachments = []
            for part in email_msg.walk():
                if part.get_content_disposition() == "attachment":
                    filename = part.get_filename()
                    content = part.get_payload(decode=True)
                    attachments.append((filename, content))

            emails.append({"subject": subject, "sender": sender, "body": body, "attachments": attachments})

            # Mark email as read
            client.add_flags(msgid, ["\\Seen"])

    return emails
