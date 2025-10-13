import cv2
import pytesseract
import requests
import pandas as pd
import spacy
import re
from PIL import Image 
import psycopg2
import os
from datetime import datetime
import json
from typing import Dict, Any, List, Optional, Tuple


# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model not found. Please install with: python -m spacy download en_core_web_sm")
    nlp = None

# Configure pytesseract path based on OS
def setup_tesseract():
    if os.name == 'nt':  # Windows
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    else:  # Linux/Mac
        pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # Updated default path

setup_tesseract()

SUPABASE_URL = "https://ryxijguahtthyxxfpkek.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ5eGlqZ3VhaHR0aHl4eGZwa2VrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjAwODUyMTksImV4cCI6MjA3NTY2MTIxOX0.9Ogoz11V59q69ZUVemUY0u59459CGBLtrEg1GERrOYQ"

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

# Enhanced text extraction
def extract_text(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Multiple preprocessing techniques
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        configs = [
            '--psm 6',  # Uniform block of text
            '--psm 4',  # Single column of text
            '--psm 3'   # Fully automatic page segmentation
        ]
        
        best_text = ""
        for config in configs:
            for preprocessed in [thresh1, thresh2]:
                pil_img = Image.fromarray(preprocessed)
                text = pytesseract.image_to_string(pil_img, config=config)
                if len(text.strip()) > len(best_text.strip()):
                    best_text = text
        
        return best_text.strip()
            
    except Exception as e:
        print(f"Error in text extraction: {e}")
        return ""

# --- Configuration ---
GOOGLE_BOOKS_API_URL = "https://www.googleapis.com/books/v1/volumes"
OPEN_LIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"
API_TIMEOUT = 10

# --- Helper Function for ISBN Extraction ---

def extract_isbn_from_text(text: str) -> Optional[str]:
    """
    Scans a large block of text for a valid ISBN-10 or ISBN-13 pattern.
    """
    # Pattern to find ISBN-13 or ISBN-10, tolerating hyphens/spaces and 'ISBN' prefix
    pattern = r'(?:ISBN(?:-1[03])?:?\s*)?((?:97[8-9]\s?[-]?\s?)?\d{1,5}\s?[-]?\s?\d{1,7}\s?[-]?\s?\d{1,6}\s?[-]?\s?[\dxX])'
    
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    for match in matches:
        cleaned_isbn = re.sub(r'[^\dXx]', '', match).upper()
        
        if len(cleaned_isbn) in [10, 13]:
            return cleaned_isbn
            
    return None

# --- Step 1: Dynamic OCR & Information Extraction ---

def extract_info_from_images(image_paths: List[str]) -> Dict[str, str]:
    """
    Performs OCR on all provided images and aggregates the text.
    The primary goal is to get a broad text dump (for title/author) and a precise ISBN.
    """
    if not image_paths:
         return {"raw_text_dump": "", "isbn": None}
         
    print(f"--- 📚 Running OCR on {len(image_paths)} images. ---")
    
    full_text_dump = []
    found_isbn = None
    
    for path in image_paths:
        raw_text = ""
        try:
            img = Image.open(path)
            raw_text = pytesseract.image_to_string(img)
        except pytesseract.TesseractNotFoundError:
            print("ERROR: Tesseract not found. Cannot perform OCR.")
            return {"raw_text_dump": "error", "isbn": None}
        except FileNotFoundError:
            print(f"ERROR: Image file not found at {path}. Skipping.")
            continue
        except Exception as e:
            print(f"ERROR during processing {path}: {e}")
            continue

        clean_text = ' '.join(raw_text.split())
        full_text_dump.append(clean_text)
        
        # Look for the ISBN in all pages (prioritizes the first valid one found)
        if not found_isbn:
            found_isbn = extract_isbn_from_text(clean_text)

    combined_text = " ".join(full_text_dump)
    print(f"✅ OCR successful. ISBN found: {found_isbn}")
    
    return {
        "raw_text_dump": combined_text,
        "isbn": found_isbn
    }

# --- Step 2 & 3: Dynamic API Lookup (ISBN Priority) ---

def search_book_apis(ocr_data: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Prioritizes ISBN search, then falls back to text search using the combined OCR dump.
    """
    isbn = ocr_data.get("isbn")
    raw_text = ocr_data.get("raw_text_dump", "")
    
    # ------------------ 2a. Primary Search: ISBN (Google Books) ------------------
    if isbn:
        print(f"\n--- 🌐 Searching Google Books (Primary: ISBN) with ISBN: {isbn} ---")
        google_params = {"q": f"isbn:{isbn}", "maxResults": 1}
        try:
            response = requests.get(GOOGLE_BOOKS_API_URL, params=google_params, timeout=API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if data.get('totalItems', 0) > 0 and 'items' in data:
                print("✅ Found match on Google Books using ISBN.")
                return data['items'], 'google'
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during ISBN Google search: {e}")

    # ------------------ 2b. Secondary Search: Full Text (Google Books) ------------------
    if raw_text and raw_text != "error":
        print(f"\n--- 🌐 Searching Google Books (Secondary: Text) with query: '{raw_text[:50]}...' ---")
        google_params = {"q": raw_text, "maxResults": 5}
        try:
            response = requests.get(GOOGLE_BOOKS_API_URL, params=google_params, timeout=API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if data.get('totalItems', 0) > 0 and 'items' in data:
                print(f"✅ Found {data['totalItems']} result(s) on Google Books using text.")
                return data['items'], 'google'
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during Text Google search: {e}. Attempting fallback...")

    # ------------------ 3a. Fallback Search: Full Text (Open Library) ------------------
    if raw_text and raw_text != "error":
        print(f"\n--- 🌐 Searching Open Library (Fallback: Text) with query: '{raw_text[:50]}...' ---")
        open_library_params = {"q": raw_text, "limit": 5}
        try:
            response = requests.get(OPEN_LIBRARY_SEARCH_URL, params=open_library_params, timeout=API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if data.get('numFound', 0) > 0 and 'docs' in data:
                print(f"✅ Found {data['numFound']} result(s) on Open Library using text.")
                return data['docs'], 'openlibrary'
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during Open Library search: {e}.")

    print("❌ No results found on any API.")
    return [], None

# --- Step 4: Dynamic Information Refinement (Unchanged) ---

def refine_info_from_api(api_results: List[Dict[str, Any]], api_source: str) -> Dict[str, str]:
    """
    Processes the API results based on the source to extract structured book information.
    """
    refined_data = {}
    if not api_results or not api_source:
        return refined_data
        
    print(f"\n--- ✅ Refining Information from Best {api_source.title()} Match ---")
    
    best_match = api_results[0] 

    if api_source == 'google':
        info = best_match.get('volumeInfo', {})
        authors = ", ".join(info.get('authors', []))
        isbn_13 = next((id_['identifier'] for id_ in info.get('industryIdentifiers', []) if id_['type'] == 'ISBN_13'), None)
        
        refined_data = {
            "title": info.get('title'),
            "subtitle": info.get('subtitle'),
            "author": authors,
            "publisher": info.get('publisher'),
            "publishedDate": info.get('publishedDate'),
            "pages": info.get('pageCount'),
            "isbn_13": isbn_13
        }

    elif api_source == 'openlibrary':
        authors = ", ".join(best_match.get('author_name', []))
        publishers = ", ". join(best_match.get('publisher', []))
        isbn_13 = next((isbn for isbn in best_match.get('isbn', []) if len(isbn) == 13), None)

        refined_data = {
            "title": best_match.get('title'),
            "author": authors,
            "publisher": publishers,
            "publishedDate": str(best_match.get('first_publish_year')),
            "edition_count": best_match.get('edition_count'),
            "isbn_13": isbn_13
        }
        
    return {k: v for k, v in refined_data.items() if v and str(v).strip()}

def extract_only_isbn(image_path):
    """
    Extract ONLY ISBN from image, ignoring all other text
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return ""
        
        # Preprocess for better number recognition
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use OCR configuration optimized for numbers and ISBN patterns
        pil_img = Image.fromarray(thresh)
        
        # Try different configurations for ISBN extraction
        configs = [
            '--psm 6 -c tessedit_char_whitelist=0123456789Xx-',  # Only numbers, X, and hyphens
            '--psm 4 -c tessedit_char_whitelist=0123456789Xx-',  # Single column
            '--psm 11'  # Sparse text
        ]
        
        for config in configs:
            text = pytesseract.image_to_string(pil_img, config=config)
            
            # Look for ISBN patterns only
            isbn_matches = re.findall(r'ISBN[-:\s]*([\d\-Xx]{10,17})', text, re.IGNORECASE)
            
            for isbn_match in isbn_matches:
                isbn_clean = re.sub(r'[^\dXx]', '', isbn_match.upper())
                if len(isbn_clean) in [10, 13]:
                    print(f"✅ Extracted ISBN: {isbn_clean}")
                    return isbn_clean
        
        # If ISBN pattern not found, look for 10/13 digit sequences
        text = pytesseract.image_to_string(pil_img, config='--psm 6')
        
        # Find all number sequences that could be ISBNs
        number_sequences = re.findall(r'\b(?:\d[\d\-]*\d)\b', text)
        
        for seq in number_sequences:
            clean_seq = re.sub(r'[^\d]', '', seq)
            if len(clean_seq) in [10, 13]:
                print(f"✅ Found potential ISBN: {clean_seq}")
                return clean_seq
        
        print("❌ No ISBN found in image")
        return ""
        
    except Exception as e:
        print(f"Error extracting ISBN: {e}")
        return ""

def extract_complete_book_info(image_paths):
    """
    Extract all required book information for database insertion
    """
    print("🚀 Extracting complete book information...")
    
    # Get OCR data and API info
    ocr_data = extract_info_from_images(image_paths)
    api_results, api_source = search_book_apis(ocr_data)
    api_data = refine_info_from_api(api_results, api_source)
    
    # Extract additional details from OCR text
    raw_text = ocr_data.get("raw_text_dump", "")
    complete_data = extract_additional_details(raw_text, api_data)
    
    return complete_data

def extract_additional_details(raw_text, api_data):
    """
    Extract additional details like Indian ISBN, contributors, etc.
    """
    # Extract Indian ISBN if available
    isbn_indian = extract_indian_isbn(raw_text)
    
    # Extract contributors/editors
    contributors = extract_contributors(raw_text)
    
    # Extract edition information
    edition = extract_edition_info(raw_text, api_data.get('edition_count'))
    
    # Extract publisher information (original vs reprint)
    publisher_info = extract_publisher_info(raw_text, api_data.get('publisher'))
    
    # Build complete data structure matching your database schema
    complete_data = {
        "title": api_data.get('Full Title') or api_data.get('title', ''),
        "author": api_data.get('author', ''),
        "edition": edition,
        "ISBN": {
            "international": api_data.get('isbn_13') or api_data.get('ocr_isbn', ''),
            "indian_reprint": isbn_indian
        },
        "publisher": {
            "original": publisher_info.get('original', api_data.get('publisher', '')),
            "reprint": publisher_info.get('reprint', '')
        },
        "published_date": api_data.get('publishedDate', ''),
        "contributors": contributors
    }
    
    print("📋 COMPLETE EXTRACTED DATA:")
    print("=" * 50)
    for key, value in complete_data.items():
        if value:
            print(f"   {key}: {value}")
    print("=" * 50)
    
    return complete_data

def extract_indian_isbn(text):
    """
    Extract Indian reprint ISBN specifically
    """
    # Look for Indian edition patterns
    if 'india' in text.lower() or 'indian' in text.lower():
        isbn_matches = re.findall(r'ISBN[-:\s]*([\d\-Xx]{10,17})', text, re.IGNORECASE)
        for match in isbn_matches:
            clean_isbn = re.sub(r'[^\dXx]', '', match.upper())
            if len(clean_isbn) in [10, 13]:
                return clean_isbn
    return ""

def extract_contributors(text):
    """
    Extract contributors like editors, etc.
    """
    contributors = []
    lines = text.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        # Look for editor information
        if any(role in line_lower for role in ['editor:', 'edited by', 'development editor']):
            # Extract name after the role
            for role in ['editor:', 'edited by', 'development editor']:
                if role in line_lower:
                    name = line.split(role)[-1].strip()
                    if name and len(name.split()) <= 4:
                        contributors.append({
                            "role": role.replace(':', '').strip(),
                            "name": name.title()
                        })
                    break
    
    return contributors if contributors else []

def extract_edition_info(text, api_edition_count):
    """
    Extract edition information
    """
    # Look for edition patterns
    edition_patterns = [
        r'(\d+(?:st|nd|rd|th)\s+[Ee]dition)',
        r'([Ff]irst|[Ss]econd|[Tt]hird|[Ff]ourth|[Ff]ifth)\s+[Ee]dition',
        r'([Nn]inth|[Tt]enth|[Ee]leventh|[Tt]welfth)\s+[Ee]dition'
    ]
    
    for pattern in edition_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).title()
    
    # Fallback to API edition count
    if api_edition_count:
        return f"{api_edition_count} Edition"
    
    return ""

def extract_publisher_info(text, api_publisher):
    """
    Extract original vs reprint publisher information
    """
    publisher_info = {
        "original": api_publisher or "",
        "reprint": ""
    }
    
    lines = text.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        # Look for Indian reprint publisher
        if 'india' in line_lower and 'pearson' in line_lower:
            publisher_info["reprint"] = "Pearson India Education Services Pvt. Ltd"
        elif 'pearson' in line_lower and not publisher_info["original"]:
            publisher_info["original"] = "Pearson Education"
    
    return publisher_info

# Database insertion functions matching your schema
def insert_complete_book_data(data):
    """
    Insert complete book data into database using your schema
    """
    try:
        print("📦 Inserting complete book data into database...")
        
        # Generate control number
        control_no = f"LBS-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # 1. Insert or get publisher (original)
        publisher_original = data.get("publisher", {}).get("original", "Unknown Publisher")
        insert_publisher(publisher_original)
        
        # 2. Insert authors
        author_name = data.get("author", "Unknown Author")
        author_ids = insert_author(author_name)
        
        # 3. Insert subject
        subject_heading = data.get("title", "Unknown Title")[:100]
        subject_id = insert_subject(subject_heading)
        
        # 4. Get publisher_id
        publisher_id = get_publisher_id(publisher_original)
        
        # 5. Insert MARC record
        record_id = insert_marc_record(control_no, publisher_id)
        
        if record_id:
            # 6. Link authors to record
            for author_id in author_ids:
                insert_record_author(record_id, author_id)
            
            # 7. Link subject to record
            if subject_id:
                insert_record_subject(record_id, subject_id)
            
            # 8. Insert MARC fields
            insert_marc_fields(record_id, data)
            
            print("✅ Book inserted successfully!")
            return True, f"Book '{data.get('title', 'Unknown')}' successfully added with Control No: {control_no}"
        
        return False, "Failed to create MARC record"
            
    except Exception as e:
        print(f"❌ Database insertion failed: {str(e)}")
        return False, f"Database insertion failed: {str(e)}"

def insert_publisher(name):
    """Insert publisher if not exists"""
    if not name or name == "Unknown Publisher":
        return
    
    try:
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/publishers",
            headers=headers,
            json={"name": name},
            params={"on_conflict": "name"}
        )
        print(f"✅ Publisher handled: {name}")
    except Exception as e:
        print(f"❌ Publisher insertion error: {e}")

def insert_author(author_name):
    """Insert author if not exists"""
    author_ids = []
    if author_name and author_name != "Unknown Author":
        try:
            response = requests.post(
                f"{SUPABASE_URL}/rest/v1/authors",
                headers=headers,
                json={"name": author_name, "role": "Author"},
                params={"on_conflict": "name"}
            )
            if response.status_code in [200, 201]:
                author_data = response.json()
                if isinstance(author_data, list) and len(author_data) > 0:
                    author_ids.append(author_data[0]['author_id'])
                    print(f"✅ Author handled: {author_name}")
        except Exception as e:
            print(f"❌ Author insertion error: {e}")
    
    return author_ids

def insert_subject(subject_heading):
    """Insert subject if not exists"""
    try:
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/subjects",
            headers=headers,
            json={
                "subject_heading": subject_heading,
                "classification_code": "GEN"  # Default classification
            },
            params={"on_conflict": "subject_heading"}
        )
        if response.status_code in [200, 201]:
            subject_data = response.json()
            if isinstance(subject_data, list) and len(subject_data) > 0:
                return subject_data[0]['subject_id']
    except Exception as e:
        print(f"❌ Subject insertion error: {e}")
    return None

def get_publisher_id(publisher_name):
    """Get publisher ID"""
    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/publishers?name=eq.{publisher_name}",
            headers=headers
        )
        if response.status_code == 200:
            publishers_data = response.json()
            if publishers_data and len(publishers_data) > 0:
                return publishers_data[0]['publisher_id']
    except Exception as e:
        print(f"❌ Get publisher ID error: {e}")
    return 1  # Default fallback

def insert_marc_record(control_no, publisher_id):
    """Insert MARC record"""
    try:
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/marc_records",
            headers=headers,
            json={
                "control_no": control_no,
                "record_type": "Book",
                "publisher_id": publisher_id
            }
        )
        if response.status_code in [200, 201]:
            marc_data = response.json()
            if isinstance(marc_data, list) and len(marc_data) > 0:
                return marc_data[0]['record_id']
    except Exception as e:
        print(f"❌ MARC record insertion error: {e}")
    return None

def insert_record_author(record_id, author_id):
    """Link record with author"""
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/record_authors",
            headers=headers,
            json={
                "record_id": record_id,
                "author_id": author_id
            }
        )
    except Exception as e:
        print(f"❌ Record-author linking error: {e}")

def insert_record_subject(record_id, subject_id):
    """Link record with subject"""
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/record_subjects",
            headers=headers,
            json={
                "record_id": record_id,
                "subject_id": subject_id
            }
        )
    except Exception as e:
        print(f"❌ Record-subject linking error: {e}")

def insert_marc_fields(record_id, data):
    """Insert all MARC fields"""
    # Title field (245)
    title = data.get("title", "")
    if title:
        insert_marc_field(record_id, "245", "00", title)
    
    # ISBN field (020) - International
    isbn_int = data.get("ISBN", {}).get("international", "")
    if isbn_int:
        insert_marc_field(record_id, "020", "00", f"ISBN {isbn_int}")
    
    # ISBN field (020) - Indian
    isbn_indian = data.get("ISBN", {}).get("indian_reprint", "")
    if isbn_indian:
        insert_marc_field(record_id, "020", "00", f"ISBN {isbn_indian} (Indian Reprint)")
    
    # Publication date (260)
    published_date = data.get("published_date", "")
    if published_date:
        insert_marc_field(record_id, "260", "00", f"© {published_date}")
    
    # Edition (250)
    edition = data.get("edition", "")
    if edition:
        insert_marc_field(record_id, "250", "00", edition)
    
    # Publisher (264)
    publisher = data.get("publisher", {}).get("original", "")
    if publisher:
        insert_marc_field(record_id, "264", "00", publisher)

def insert_marc_field(record_id, tag, indicators, field_value):
    """Insert individual MARC field"""
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/marc_fields",
            headers=headers,
            json={
                "record_id": record_id,
                "tag": tag,
                "indicators": indicators,
                "field_value": field_value
            }
        )
    except Exception as e:
        print(f"❌ MARC field insertion error: {e}")

    
def process_two_images_and_save(image1_path, image2_path):
    """
    Main function that processes images and saves complete data to database
    """
    try:
        # Extract complete book information using BOTH images
        image_paths = [image1_path, image2_path]
        complete_data = extract_complete_book_info(image_paths)
        
        # Insert into database using new insertion method
        success, message = insert_complete_book_data(complete_data)
        
        return success, message, complete_data
        
    except Exception as e:
        error_msg = f"Error processing images: {str(e)}"
        print(f"❌ {error_msg}")
        return False, error_msg, None
