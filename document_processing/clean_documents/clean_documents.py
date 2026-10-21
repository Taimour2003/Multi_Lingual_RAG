import re


def cleaning_documents(docs):
    # 1. Normalize whitespace (replace multiple spaces/tabs with a single space)
    raw_text = docs
    text = re.sub(r"[ \t]+", " ", raw_text)

    # 2. Normalize multiple newlines (e.g., 3 or more blank lines) down to double newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    
    cleaned_docs = text.replace("IELTS Preparation Guide — English Edition", "")
    # print("Cleaned Document ", cleaned_docs)
    return cleaned_docs.strip()
