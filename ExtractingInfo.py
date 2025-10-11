import cv2
import pytesseract
import requests
import pandas as pd
import spacy
import re
from PIL import Image 

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Configure pytesseract (adjust for your system)
pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"  # Mac example

# Function: Extract text from an image
def extract_text(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    # ✅ Convert OpenCV image (numpy array) to PIL image
    pil_img = Image.fromarray(gray)

    # ✅ Now pytesseract can read it safely
    text = pytesseract.image_to_string(pil_img)
    return text.strip()

# Function: Use NLP to extract entities
def parse_book_info(text):
    doc = nlp(text)
    entities = {"title": [], "authors": [], "publishers": []}

    for ent in doc.ents:
        if ent.label_ == "WORK_OF_ART":
            entities["title"].append(ent.text)
        elif ent.label_ == "PERSON":
            entities["authors"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["publishers"].append(ent.text)

    # Fallback regex rules
    for line in text.split("\n"):
        if re.search(r"\bby\b", line, re.I):
            entities["authors"].append(line.replace("by", "").strip())
        if re.search(r"(press|publisher|media)", line, re.I):
            entities["publishers"].append(line.strip())

    return entities

# Function: Get details from Google Books API
def fetch_from_google_books(title):
    url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{title}"
    response = requests.get(url).json()
    if "items" in response:
        book = response["items"][0]["volumeInfo"]
        return {
            "title": book.get("title", ""),
            "authors": ", ".join(book.get("authors", [])),
            "publisher": book.get("publisher", ""),
            "publishedDate": book.get("publishedDate", ""),
            "ISBN": ", ".join([i["identifier"] for i in book.get("industryIdentifiers", [])]) if "industryIdentifiers" in book else ""
        }
    return None

# Combine data from both images
def process_two_images(image1, image2):
    text1 = extract_text(image1)
    text2 = extract_text(image2)
    combined_text = text1 + "\n" + text2

    entities = parse_book_info(combined_text)
    print("Extracted (Raw):", entities)

    # Try fetching from Google Books API using first title
    if entities["title"]:
        book_data = fetch_from_google_books(entities["title"][0])
        if book_data:
            print("Fetched from Google Books:", book_data)
            return book_data

    # If API fails, fallback to manual extraction
    return {
        "title": entities["title"][0] if entities["title"] else "Unknown",
        "authors": ", ".join(entities["authors"]) if entities["authors"] else "Unknown",
        "publisher": ", ".join(entities["publishers"]) if entities["publishers"] else "Unknown",
        "publishedDate": "Unknown",
        "ISBN": "Unknown"
    }

# Save data to Excel
# def save_to_excel(data, filename="books.xlsx"):
#     df = pd.DataFrame([data])
#     df.to_excel(filename, index=False)
#     print(f"Data saved to {filename}")

# Main workflow
if __name__ == "__main__":
    image1 = "image 1.jpg"
    image2 = "image 2.jpg"

    book_info = process_two_images(image1, image2)
    #save_to_excel(book_info)
