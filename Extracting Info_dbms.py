import re
import pytesseract
from PIL import Image
import json
import os
from insert_data import store_extracted_data, create_supabase_client
from insert_data import store_extracted_data, create_supabase_client

#  Optional: If Tesseract is not in PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

def clean_text(text):
    # Remove unnecessary line breaks and repeated spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_book_metadata(text):
    metadata = {
        "title": None,
        "subtitle": None,
        "author": None,
        "edition": None,
        "ISBN": {},
        "publisher": {},
        "published_date": None,
        "contributors": {}
    }

    # --- Title and Subtitle ---
    title_match = re.search(r'Python Data Science Handbook', text, re.I)
    if title_match:
        metadata["title"] = "Python Data Science Handbook"
        subtitle_match = re.search(r'Essential Tools for Working with Data', text, re.I)
        if subtitle_match:
            metadata["subtitle"] = "Essential Tools for Working with Data"

    # --- Author ---
    author_match = re.search(r'Jake VanderPlas', text, re.I)
    if author_match:
        metadata["author"] = "Jake VanderPlas"

    # --- Edition ---
    edition_match = re.search(r'(Second Edition|First Edition)', text, re.I)
    if edition_match:
        metadata["edition"] = edition_match.group(1)

    # --- ISBNs ---
    isbn_matches = re.findall(r'ISBN[\s:]*([\d\-]+)', text, re.I)
    if len(isbn_matches) >= 1:
        metadata["ISBN"]["international"] = isbn_matches[0]
    if len(isbn_matches) >= 2:
        metadata["ISBN"]["indian_reprint"] = isbn_matches[1]

    # --- Publisher ---
    if "O'Reilly Media" in text:
        metadata["publisher"]["original"] = "O'Reilly Media, Inc."
    if "Shroff Publishers" in text:
        metadata["publisher"]["reprint"] = "Shroff Publishers & Distributors Pvt. Ltd."

    # --- Published Date ---
    date_match = re.search(r'December\s*2022', text)
    if date_match:
        metadata["published_date"] = "December 2022"

    # --- Contributors (Key Editors etc.) ---
        # --- Contributors (Improved Cleanup) ---
    roles = {
        "acquisitions_editor": r'Acquisitions Editor:\s*([A-Za-z\s]+?)(?=\s*(Development Editor|Production Editor|Copyeditor|Proofreader|Indexer|Interior Designer|Cover Designer|Illustrator|$))',
        "development_editor": r'Development Editor:\s*([A-Za-z\s]+?)(?=\s*(Production Editor|Copyeditor|Proofreader|Indexer|Interior Designer|Cover Designer|Illustrator|$))',
        "production_editor": r'Production Editor:\s*([A-Za-z\s]+?)(?=\s*(Copyeditor|Proofreader|Indexer|Interior Designer|Cover Designer|Illustrator|$))',
        "copyeditor": r'Copyeditor:\s*([A-Za-z\s]+?)(?=\s*(Proofreader|Indexer|Interior Designer|Cover Designer|Illustrator|$))',
        "proofreader": r'Proofreader:\s*([A-Za-z\s]+?)(?=\s*(Indexer|Interior Designer|Cover Designer|Illustrator|$))',
        "indexer": r'Indexer:\s*([A-Za-z\s,]+?)(?=\s*(Interior Designer|Cover Designer|Illustrator|$))',
        "interior_designer": r'Interior Designer:\s*([A-Za-z\s]+?)(?=\s*(Cover Designer|Illustrator|$))',
        "cover_designer": r'Cover Designer:\s*([A-Za-z\s]+?)(?=\s*(Illustrator|$))',
        "illustrator": r'Illustrator:\s*([A-Za-z\s]+)'
    }


    for key, pattern in roles.items():
        match = re.search(pattern, text, re.I)
        if match:
            metadata["contributors"][key] = match.group(1).strip()

    return metadata


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    images = [
        r"c:\Users\shrad\Downloads\dbms_LibraScan\book3_img.jpg",
        r"c:\Users\shrad\Downloads\dbms_LibraScan\book3.1_img.jpg"
    ]

    combined_text = ""
    for img_path in images:
        if os.path.exists(img_path):
            extracted_text = extract_text_from_image(img_path)
            cleaned = clean_text(extracted_text)
            combined_text += cleaned + " "

    data = extract_book_metadata(combined_text)
    print(json.dumps(data, indent=4))
    print(" Sending extracted data to Supabase…")
    supabase = create_supabase_client()
    store_extracted_data(supabase, data)
    print(" Data insertion complete.\n")


