import cv2
import pytz
import pytesseract
import requests
import pandas as pd
import spacy
import re
from PIL import Image 
import psycopg2
import os
from openai import OpenAI
from datetime import datetime
import json
import google.generativeai as genai
from typing import Dict, Any, List, Optional, Tuple


# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model not found. Please install with: python -m spacy download en_core_web_sm")
    nlp = None

# Configure pytesseract path based on OS
def setup_tesseract():
   pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract' 

setup_tesseract()

SUPABASE_URL = "https://ryxijguahtthyxxfpkek.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ5eGlqZ3VhaHR0aHl4eGZwa2VrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjAwODUyMTksImV4cCI6MjA3NTY2MTIxOX0.9Ogoz11V59q69ZUVemUY0u59459CGBLtrEg1GERrOYQ"

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

#Function to remove unwanted characters
def remove_whitespace(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def get_india_time():
    """Return the current time in India (GMT+5:30) as ISO string."""
    india_tz = pytz.timezone("Asia/Kolkata")
    return datetime.now(india_tz).isoformat()

def clean_text(value, max_len=255, allow_punct=False):
    """Basic cleaning of short text fields (names, publisher names)."""
    if not value:
        return None
    v = str(value)
    v = remove_whitespace(v)
    if not allow_punct:
        v = re.sub(r'[^A-Za-z0-9&.,:\'"\-\s()]', '', v)
    else:
        v = re.sub(r'[^\x00-\x7F]', '', v)
    v = v.strip()
    return v[:max_len] if max_len else v

def clean_publisher(raw_text):
    """Return the best candidate publisher name from noisy text."""
    if not raw_text:
        return None
    s = str(raw_text)
    parts = re.split(r'[,|\n]+', s)
    keywords = ['media', 'press', 'publisher', 'publications', 'publishing', 'house', 'edition', 'inc', "ltd"]
    for p in parts:
        if any(k in p.lower() for k in keywords):
            return clean_text(p)
    for p in parts:
        p_clean = clean_text(p)
        if p_clean and len(p_clean.split()) <= 6 and not re.search(r'copyright|protected|reprint|all rights', p_clean, re.I):
            return p_clean
    return clean_text(parts[0]) if parts else clean_text(s)

def clean_authors(raw_text):
    """Return a list of probable author names cleaned and deduped."""
    if not raw_text:
        return []
    if isinstance(raw_text, str):
        candidates = re.split(r'[,|\n]+', raw_text)
    elif isinstance(raw_text, (list, tuple)):
        candidates = raw_text
    else:
        candidates = [str(raw_text)]

    invalid_words = {
        "production", "development", "responsibility", "illustrator",
        "release", "reprint", "edition", "media", "press", "publication",
        "publisher", "copyright", "inc", "oreilly", "limited", "rights",
        "isbn", "published", "protected", "form", "means", "photocopying",
        "handbook", "first", "second", "release", "revision", "history"
    }

    cleaned = []
    for c in candidates:
        c = clean_text(c)
        if not c:
            continue
        if any(w in c.lower() for w in invalid_words):
            continue
        if any(char.isdigit() for char in c):
            continue
        words = c.split()
        if not (1 <= len(words) <= 4):
            continue
        if any(len(w) <= 1 for w in words):
            continue
        cleaned.append(c)

    seen = set()
    result = []
    for name in cleaned:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            result.append(name)
    return result

def clean_title(raw_title, raw_publisher=None, raw_authors=None):
    """Try to return a clean title."""
    if raw_title and isinstance(raw_title, str) and raw_title.strip() and raw_title.lower() != 'unknown':
        return clean_text(raw_title, max_len=1000, allow_punct=True)

    if raw_publisher:
        parts = re.split(r'[,|\n]+', str(raw_publisher))
        for p in parts:
            p_clean = clean_text(p, max_len=1000, allow_punct=True)
            if p_clean and 2 <= len(p_clean.split()) <= 10 and not re.search(r'publisher|media|press|inc|copyright|reprint', p_clean, re.I):
                return p_clean

    if raw_authors and isinstance(raw_authors, str):
        parts = raw_authors.split('\n')
        if len(parts) >= 2:
            candidate = clean_text(parts[0], max_len=1000, allow_punct=True)
            if candidate and 2 <= len(candidate.split()) <= 10:
                return candidate

    return "Unknown"

def extract_year(raw):
    """Return the first plausible 4-digit year (1500-2099)."""
    if not raw:
        return None
    s = str(raw)
    m = re.search(r'\b(1[5-9]\d{2}|20[0-9]{2})\b', s)
    if m:
        return m.group(0)
    return None

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
         
    print(f"--- Running OCR on {len(image_paths)} images. ---")
    
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

# --- Step 2 & 3: API Lookup (ISBN Priority) ---
def search_book_apis(ocr_data: Dict[str, str], indian_isbn: str = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Prioritizes ISBN search (both international and Indian), then falls back to text search.
    """
    isbn = ocr_data.get("isbn")
    raw_text = ocr_data.get("raw_text_dump", "")
    
    # ------------------ 2a. Primary Search: International ISBN (Google Books) ------------------
    if isbn:
        print(f"\n--- 🌐 Searching Google Books (Primary: International ISBN) with ISBN: {isbn} ---")
        google_params = {"q": f"isbn:{isbn}", "maxResults": 1}
        try:
            response = requests.get(GOOGLE_BOOKS_API_URL, params=google_params, timeout=API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if data.get('totalItems', 0) > 0 and 'items' in data:
                print("✅ Found match on Google Books using International ISBN.")
                return data['items'], 'google'
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during International ISBN Google search: {e}")

    # ------------------ 2b. Secondary Search: Indian ISBN (Google Books) ------------------
    if indian_isbn:
        print(f"\n--- 🌐 Searching Google Books (Secondary: Indian ISBN) with ISBN: {indian_isbn} ---")
        google_params = {"q": f"isbn:{indian_isbn}", "maxResults": 1}
        try:
            response = requests.get(GOOGLE_BOOKS_API_URL, params=google_params, timeout=API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if data.get('totalItems', 0) > 0 and 'items' in data:
                print("✅ Found match on Google Books using Indian ISBN.")
                return data['items'], 'google'
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during Indian ISBN Google search: {e}")

    # ------------------ 2c. Tertiary Search: Full Text (Google Books) ------------------
    if raw_text and raw_text != "error":
        print(f"\n--- 🌐 Searching Google Books (Tertiary: Text) with query: '{raw_text[:50]}...' ---")
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
        print(f"\n--- Searching Open Library (Fallback: Text) with query: '{raw_text[:50]}...' ---")
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

# --- Step 4: Dynamic Information Refinement ---

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
    Extract all required book information for database insertion,
    including intelligent subject detection using Google Books or Gemini AI.
    """
    print("Extracting complete book information...")
    
    # Step 1: OCR and API extraction
    ocr_data = extract_info_from_images(image_paths)
    
    # Step 1.5: Extract Indian ISBN early for API searching
    raw_text = ocr_data.get("raw_text_dump", "")
    indian_isbn = extract_indian_isbn(raw_text)
    
    # Step 2: Search APIs with BOTH ISBNs
    api_results, api_source = search_book_apis(ocr_data, indian_isbn)
    api_data = refine_info_from_api(api_results, api_source)
    
    # Rest of the function remains the same...
    complete_data = extract_additional_details(raw_text, api_data)

    # Step 3: --- SUBJECT DETECTION ---
    try:
        subjects = []

        # 1️⃣ Try Google Books categories first (if present)
        if api_results and len(api_results) > 0:
            best_match = api_results[0]
            if api_source == 'google':
                volume_info = best_match.get("volumeInfo", {})
                categories = volume_info.get("categories", [])
            elif api_source == 'openlibrary':
                # OpenLibrary uses subject_key instead of categories
                categories = best_match.get("subject_key", [])
            else:
                categories = []
            
            if categories:
                subjects = [cat.strip() for cat in categories if cat.strip()]
                print(f"✅ Subjects from {api_source}: {subjects}")

        # 2️⃣ Fallback to Gemini AI subject generation
        if not subjects:
            print("⚙️ No subjects found via API — generating via Gemini AI...")
            title = complete_data.get("title", "")
            author = complete_data.get("author", "")
            
            # Get description from API results if available
            description = ""
            if api_results and len(api_results) > 0:
                best_match = api_results[0]
                if api_source == 'google':
                    description = best_match.get("volumeInfo", {}).get("description", "")
                elif api_source == 'openlibrary':
                    description = best_match.get("first_sentence", [""])[0] if best_match.get("first_sentence") else ""
            
            subjects = generate_subjects_with_gemini(title, author, description)

        # 3️⃣ Final fallback if Gemini also fails
        if not subjects:
            print("⚠️ No subjects generated, using default")
            subjects = ["General"]

        # 4️⃣ Finalize subject assignment
        subject_str = ", ".join(subjects[:5])  # Limit to 5 subjects max
        complete_data["subjects"] = subject_str
        complete_data["subjects_list"] = subjects  # Keep list for database insertion
        print(f"📚 Final subjects assigned: {subject_str}")

    except Exception as e:
        print(f"⚠️ Subject generation error: {e}")
        complete_data["subjects"] = "General"
        complete_data["subjects_list"] = ["General"]

    # Step 4: Return enriched data
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
    
    # Clean data using new cleaning functions
    cleaned_title = clean_title(api_data.get('title'), publisher_info.get('original'), api_data.get('author'))
    cleaned_authors = clean_authors(api_data.get('author'))
    cleaned_publisher = clean_publisher(publisher_info.get('original'))
    pub_year = extract_year(api_data.get('publishedDate'))
    
    # Build complete data structure matching your database schema
    complete_data = {
        "title": cleaned_title,
        "author": ", ".join(cleaned_authors) if cleaned_authors else "",
        "authors_list": cleaned_authors,  # Store cleaned list for insertion
        "edition": edition,
        "ISBN": {
            "international": api_data.get('isbn_13') or extract_isbn_from_text(raw_text),
            "indian_reprint": isbn_indian
        },
        "publisher": {
            "original": cleaned_publisher,
            "reprint": publisher_info.get('reprint', '')
        },
        "published_date": pub_year,
        "contributors": contributors
    }
    
    print("COMPLETE EXTRACTED DATA:")
    print("=" * 50)
    for key, value in complete_data.items():
        if value and key != "authors_list":  # Don't print internal list
            print(f"   {key}: {value}")
    print("=" * 50)
    
    return complete_data

def extract_indian_isbn(text):
    """
    Extract Indian reprint ISBN specifically
    """
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
        if any(role in line_lower for role in ['editor:', 'edited by', 'development editor']):
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
    edition_patterns = [
        r'(\d+(?:st|nd|rd|th)\s+[Ee]dition)',
        r'([Ff]irst|[Ss]econd|[Tt]hird|[Ff]ourth|[Ff]ifth)\s+[Ee]dition',
        r'([Nn]inth|[Tt]enth|[Ee]leventh|[Tt]welfth)\s+[Ee]dition'
    ]
    
    for pattern in edition_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).title()
    
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
        if 'india' in line_lower and 'pearson' in line_lower:
            publisher_info["reprint"] = "Pearson India Education Services Pvt. Ltd"
        elif 'pearson' in line_lower and not publisher_info["original"]:
            publisher_info["original"] = "Pearson Education"
    
    return publisher_info

def insert_complete_book_data(data):
    """
    Insert complete book data into database
    """
    try:
        print("Inserting complete book data into database...")
        
        # Generate control number (use ISBN if available)
        isbn_int = data.get("ISBN", {}).get("international")
        control_no = isbn_int if isbn_int else f"LBS-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # 1. Get or create publisher with improved logic
        publisher_original = data.get("publisher", {}).get("original", "Unknown Publisher")
        publisher_id = get_or_create_publisher(publisher_original)
        
        # FIX: Continue even if publisher creation fails
        if not publisher_id:
            print("⚠️ Failed to get/create publisher, using default publisher")
            # Create a default publisher to ensure book insertion continues
            publisher_id = get_or_create_publisher("Unknown Publisher")
            if not publisher_id:
                # Last resort: create a generic publisher
                publisher_id = create_fallback_publisher()
        
        # 2. Get or create authors with improved logic
        authors_list = data.get("authors_list", [])
        if not authors_list and data.get("author"):
            authors_list = [data.get("author")]
        
        author_ids = []
        for author_name in authors_list:
            author_id = get_or_create_author(author_name)
            if author_id:
                author_ids.append(author_id)
        
        if not author_ids:
            author_id = get_or_create_author("Unknown Author")
            if author_id:
                author_ids.append(author_id)
        
        # 3. Get or create subjects (NEW - using actual subjects from data)
        subjects_list = data.get("subjects_list", ["General"])
        subject_ids = []
        
        for subject_heading in subjects_list[:3]:  # Limit to 3 subjects max
            subject_id = get_or_create_subject(subject_heading)
            if subject_id:
                subject_ids.append(subject_id)
        
        # If no subjects were created, create a default one
        if not subject_ids:
            subject_id = get_or_create_subject("General")
            if subject_id:
                subject_ids.append(subject_id)
        
        # 4. Get or create MARC record
        record_id = get_or_create_record(control_no, publisher_id)
        
        if record_id:
            # 5. Link authors to record
            for author_id in author_ids:
                link_record_author(record_id, author_id)
            
            # 6. Link subjects to record (NEW - link all subjects)
            for subject_id in subject_ids:
                link_record_subject(record_id, subject_id)
            
            # 7. Insert MARC fields with improved data
            insert_marc_fields_improved(record_id, data)
            
            print("✅ Book inserted successfully!")
            return True, f"Book '{data.get('title', 'Unknown')}' successfully added with Control No: {control_no}"
        
        return False, "Failed to create MARC record"
            
    except Exception as e:
        print(f"❌ Database insertion failed: {str(e)}")
        return False, f"Database insertion failed: {str(e)}"

def create_fallback_publisher():
    """Create a fallback publisher if all else fails"""
    try:
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/publishers",
            headers=headers,
            json={"name": "Unknown Publisher"}
        )
        if response.status_code in [200, 201]:
            publisher_data = response.json()
            if isinstance(publisher_data, list) and len(publisher_data) > 0:
                print("✅ Created fallback publisher")
                return publisher_data[0]['publisher_id']
    except Exception as e:
        print(f"❌ Failed to create fallback publisher: {e}")
    return None

def get_or_create_publisher(name):
    """Get existing publisher or create new one"""
    if not name:
        print("❌ Invalid publisher name")
        return None
    
    try:
        print(f"Attempting to get/create publisher: '{name}'")
        
        # Use ilike query with URL encoding to handle special characters
        import urllib.parse
        encoded_name = urllib.parse.quote(name)
        
        # Use ilike for case-insensitive partial matching
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/publishers?name=ilike.{encoded_name}",
            headers=headers
        )
        
        print(f"GET Response status: {response.status_code}")
        
        if response.status_code == 200:
            publishers_data = response.json()
            print(f"GET Response data: {publishers_data}")
            
            if publishers_data and len(publishers_data) > 0:
                print(f"✅ Publisher found: {name}")
                return publishers_data[0]['publisher_id']
            else:
                print("No existing publisher found, creating new one...")
                
                # Try creating new publisher
                response = requests.post(
                    f"{SUPABASE_URL}/rest/v1/publishers",
                    headers=headers,
                    json={"name": name}
                )
                
                if response.status_code == 409:
                    # Publisher exists but wasn't found by query - try exact match with different encoding
                    print("Publisher exists but query didn't find it, trying exact match...")
                    exact_response = requests.get(
                        f"{SUPABASE_URL}/rest/v1/publishers",
                        headers=headers
                    )
                    if exact_response.status_code == 200:
                        all_publishers = exact_response.json()
                        for publisher in all_publishers:
                            if publisher['name'] == name:
                                print(f"✅ Found publisher in full list: {name}")
                                return publisher['publisher_id']
                
                elif response.status_code in [200, 201]:
                    publisher_data = response.json()
                    if isinstance(publisher_data, list) and len(publisher_data) > 0:
                        print(f"✅ Publisher created: {name}")
                        return publisher_data[0]['publisher_id']
                
        print(f"❌ Could not resolve publisher: {name}")
                
    except Exception as e:
        print(f"❌ Publisher get/create error: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def get_or_create_author(author_name):
    """Get existing author or create new one"""
    if not author_name or author_name == "Unknown Author":
        return None
    
    try:
        # Check if author exists
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/authors?name=eq.{author_name}",
            headers=headers
        )
        if response.status_code == 200:
            authors_data = response.json()
            if authors_data and len(authors_data) > 0:
                print(f"✅ Author found: {author_name}")
                return authors_data[0]['author_id']
        
        # Create new author
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/authors",
            headers=headers,
            json={"name": author_name, "role": "Author"}
        )
        if response.status_code in [200, 201]:
            author_data = response.json()
            if isinstance(author_data, list) and len(author_data) > 0:
                print(f"✅ Author created: {author_name}")
                return author_data[0]['author_id']
                
    except Exception as e:
        print(f"❌ Author get/create error: {e}")
    
    return None

def get_or_create_subject(subject_heading):
    """Get existing subject or create new one"""
    try:
        # Check if subject exists
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/subjects?subject_heading=eq.{subject_heading}",
            headers=headers
        )
        if response.status_code == 200:
            subjects_data = response.json()
            if subjects_data and len(subjects_data) > 0:
                return subjects_data[0]['subject_id']
        
        # Create new subject
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/subjects",
            headers=headers,
            json={
                "subject_heading": subject_heading,
                "classification_code": "GEN"
            }
        )
        if response.status_code in [200, 201]:
            subject_data = response.json()
            if isinstance(subject_data, list) and len(subject_data) > 0:
                return subject_data[0]['subject_id']
    except Exception as e:
        print(f"❌ Subject get/create error: {e}")
    
    return None

def get_or_create_record(control_no, publisher_id):
    """Get existing record or create new one"""
    try:
        # Check if record exists (by control_no/ISBN)
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/marc_records?control_no=eq.{control_no}",
            headers=headers
        )
        if response.status_code == 200:
            records_data = response.json()
            if records_data and len(records_data) > 0:
                print(f"✅ Record found: {control_no}")
                return records_data[0]['record_id']
        
        # Create new record with initial book_count = 1
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/marc_records",
            headers=headers,
            json={
                "control_no": control_no,
                "record_type": "Book",
                "publisher_id": publisher_id,
                "book_count": 1,  # Initialize count to 1 for new books
                "created_at": get_india_time()
            }
        )
        if response.status_code in [200, 201]:
            record_data = response.json()
            if isinstance(record_data, list) and len(record_data) > 0:
                print(f"✅ Record created: {control_no}")
                return record_data[0]['record_id']
                
    except Exception as e:
        print(f"❌ Record get/create error: {e}")
    
    return None

def link_record_author(record_id, author_id):
    """Link record with author (only if not already linked)"""
    try:
        # Check if link already exists
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/record_authors?record_id=eq.{record_id}&author_id=eq.{author_id}",
            headers=headers
        )
        if response.status_code == 200:
            links_data = response.json()
            if links_data and len(links_data) > 0:
                return  # Link already exists
        
        # Create new link
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

def link_record_subject(record_id, subject_id):
    """Link record with subject (only if not already linked)"""
    try:
        # Check if link already exists
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/record_subjects?record_id=eq.{record_id}&subject_id=eq.{subject_id}",
            headers=headers
        )
        if response.status_code == 200:
            links_data = response.json()
            if links_data and len(links_data) > 0:
                return  # Link already exists
        
        # Create new link
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

def insert_marc_fields_improved(record_id, data):
    """Insert all MARC fields with proper subfield codes"""

    # Define MARC → subfield code mapping
    MARC_SUBFIELD_CODES = {
        "245": "a",  # Title
        "100": "a",  # Main Author
        "020": "a",  # ISBN
        "250": "a",  # Edition
        "260": "c",  # Date of publication
        "264": "b"   # Publisher name
    }

    def insert_field(tag, value):
        """Helper to insert field + subfield"""
        if not value:
            return
        subfield_code = MARC_SUBFIELD_CODES.get(tag, "a")
        insert_marc_field(record_id, tag, "00", value, subfield_code)

    # --- Insert all relevant fields ---
    title = data.get("title", "")
    if title:
        insert_field("245", title)

    authors_list = data.get("authors_list", [])
    if authors_list:
        insert_field("100", authors_list[0])

    isbn_int = data.get("ISBN", {}).get("international", "")
    if isbn_int:
        insert_field("020", f"ISBN {isbn_int}")

    isbn_indian = data.get("ISBN", {}).get("indian_reprint", "")
    if isbn_indian:
        insert_field("020", f"ISBN {isbn_indian} (Indian Reprint)")

    published_date = data.get("published_date", "")
    if published_date:
        insert_field("260", f"© {published_date}")

    edition = data.get("edition", "")
    if edition:
        insert_field("250", edition)

    publisher = data.get("publisher", {}).get("original", "")
    if publisher:
        insert_field("264", publisher)


def insert_marc_subfield(field_id, subfield_code, subfield_value):
    """Insert a subfield for a given MARC field"""
    try:
        subfield_payload = {
            "field_id": field_id,
            "code": subfield_code,       # correct key name
            "value": subfield_value      # correct key name
        }
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/marc_subfields",
            headers=headers,
            json=subfield_payload
        )

        if response.status_code not in [200, 201]:
            print(f"⚠️ Subfield insertion failed for field_id {field_id}: {response.text}")
        else:
            print(f"✅ Inserted subfield {subfield_code} for field_id {field_id}")

    except Exception as e:
        print(f"❌ Subfield insertion error: {e}")


def insert_marc_field(record_id, tag, indicators, field_value, subfield_code="a"):
    """Insert individual MARC field and automatically create its subfield"""
    try:
        # Step 1: Insert MARC field
        payload = {
            "record_id": record_id,
            "tag": tag,
            "indicators": indicators,
            "field_value": field_value
        }

        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/marc_fields",
            headers=headers,
            json=payload
        )

        if response.status_code not in [200, 201]:
            print(f"⚠️ MARC field insertion failed: {response.text}")
            return

        field_data = response.json()
        if isinstance(field_data, list) and len(field_data) > 0:
            field_id = field_data[0].get("field_id") or field_data[0].get("id")

            # Step 2: Insert proper subfield code (not just "a")
            if field_id:
                insert_marc_subfield(field_id, subfield_code, field_value)
                print(f"✅ Inserted MARC field {tag} with subfield ${subfield_code}.")
            else:
                print("⚠️ No field_id returned; cannot insert subfield.")
        else:
            print("⚠️ Empty response for MARC field insert.")
    except Exception as e:
        print(f"❌ MARC field insertion error: {e}")

def check_book_exists(isbn):
    """
    Check if a book already exists in the marc_records table 
    by matching the normalized ISBN (digits only) against the control_no.
    Returns a dictionary of basic book details if found, otherwise None.
    """
    try:
        if not isbn:
            print("❌ No ISBN provided for duplicate check.")
            return None

        # Normalize ISBN same way as control_no
        normalized_isbn = re.sub(r"[^0-9Xx]", "", str(isbn)).strip()
        print(f"\n--- 🔍 Checking for existing book with control_no: {normalized_isbn} ---")

        # Directly query marc_records for the control_no
        url = f"{SUPABASE_URL}/rest/v1/marc_records?control_no=eq.{normalized_isbn}&select=record_id,control_no,book_count"
        print(f"Query URL: {url}")
        
        response = requests.get(url, headers=headers)
        print(f"Response status: {response.status_code}")

        if response.status_code != 200:
            print(f"❌ Supabase error while checking duplicates: {response.status_code}")
            return None

        records = response.json()
        print(f"Records found: {len(records)}")
        
        if not records:
            print("⚠️ No existing record found for this ISBN/control_no.")
            return None

        # Found an existing record
        record = records[0]
        record_id = record.get("record_id")
        book_count = record.get("book_count", 1)
        control_no = record.get("control_no")

        print(f"✅ Duplicate found! Record ID: {record_id}, Control No: {control_no}, Count: {book_count}")

        return {
            "record_id": record_id,
            "control_no": control_no,
            "title": "Existing Book",   # You might want to fetch the actual title
            "count": book_count,
            "isbn": normalized_isbn
        }

    except Exception as e:
        print(f"❌ Error in check_book_exists: {e}")
        import traceback
        traceback.print_exc()
        return None


def update_book_count(control_no):
    """
    Increment the book count for an existing record
    """
    try:
        # First get current count
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/marc_records?control_no=eq.{control_no}",
            headers=headers
        )
        
        if response.status_code == 200:
            records = response.json()
            if records and len(records) > 0:
                current_count = records[0].get('book_count', 1)
                new_count = current_count + 1
                
                # Update the count
                update_response = requests.patch(
                    f"{SUPABASE_URL}/rest/v1/marc_records?control_no=eq.{control_no}",
                    headers=headers,
                    json={"book_count": new_count}
                )
                
                if update_response.status_code in [200, 204]:
                    print(f"✅ Book count updated from {current_count} to {new_count} for {control_no}")
                    return True, f"Book count increased to {new_count}"
                else:
                    print(f"❌ Failed to update book count: {update_response.text}")
                    return False, "Failed to update book count"
        
        return False, "Book record not found"
        
    except Exception as e:
        print(f"Error updating book count: {e}")
        return False, f"Error updating book count: {str(e)}"

def extract_isbn_for_check(image_path_1, image_path_2):
    """
    Quickly extract ISBN from the two uploaded images for duplicate checking.
    Uses the same logic as extract_text + extract_isbn_from_text but without saving anything.
    Returns the normalized ISBN string if found, otherwise None.
    """
    try:
        print("\n--- Running quick ISBN extraction for duplicate check ---")

        # Reuse your OCR extraction
        text1 = extract_text(image_path_1)
        text2 = extract_text(image_path_2)

        combined_text = f"{text1} {text2}".strip()

        if not combined_text:
            print("⚠️ No text detected from either image.")
            return None

        isbn = extract_isbn_from_text(combined_text)

        if isbn:
            clean_isbn = re.sub(r"[^0-9Xx]", "", str(isbn))
            print(f"✅ Quick ISBN extraction success: {clean_isbn}")
            return clean_isbn

        print("⚠️ No ISBN detected during quick check.")
        return None

    except Exception as e:
        print(f"❌ Error in extract_isbn_for_check: {e}")
        return None


def process_two_images_and_save(image1_path, image2_path):
    """
    Main function that processes images and saves complete data to database.
    Runs duplicate check (by control_no / ISBN) using BOTH ISBNs BEFORE inserting.
    """
    try:
        print("\n--- Running OCR on 2 images for processing ---")

        # Extract complete book information first
        image_paths = [image1_path, image2_path]
        complete_data = extract_complete_book_info(image_paths)

        # Check for duplicates using BOTH ISBNs
        isbn_int = complete_data.get("ISBN", {}).get("international")
        isbn_indian = complete_data.get("ISBN", {}).get("indian_reprint")
        
        # Try international ISBN first
        if isbn_int:
            normalized_isbn = re.sub(r"[^0-9Xx]", "", str(isbn_int))
            print(f"🔍 Checking for existing book with control_no: {normalized_isbn}")
            existing_book = check_book_exists(normalized_isbn)
            if existing_book:
                print(f"⚠️ Duplicate found using International ISBN: {normalized_isbn}. Skipping insertion.")
                return False, "Book already exists. Awaiting user decision.", {"existing_book": existing_book}
        
        # Try Indian ISBN as fallback
        if isbn_indian and not existing_book:
            normalized_isbn = re.sub(r"[^0-9Xx]", "", str(isbn_indian))
            print(f"🔍 Checking for existing book with Indian ISBN control_no: {normalized_isbn}")
            existing_book = check_book_exists(normalized_isbn)
            if existing_book:
                print(f"⚠️ Duplicate found using Indian ISBN: {normalized_isbn}. Skipping insertion.")
                return False, "Book already exists. Awaiting user decision.", {"existing_book": existing_book}

        # If no duplicate, insert new record
        print("✅ No duplicate found — proceeding with insertion...")
        success, message = insert_complete_book_data(complete_data)
        return success, message, complete_data

    except Exception as e:
        error_msg = f"Error processing images: {str(e)}"
        print(f"❌ {error_msg}")
        return False, error_msg, None
    
def generate_subjects_with_gemini(title, author="", description=""):
    """
    Generates relevant subjects for a book using Google's Gemini 2.5 Flash model.
    """
    try:
        genai.configure(api_key="AIzaSyBnBt8VF41cfzTPClI-W7UeIN95KGULC4k")

        prompt = f"""
        You are a library cataloging assistant. 
        Suggest 3–5 relevant subjects or genres for the book titled "{title}" 
        by {author}. Use concise subject names like 'Hinduism', 'Religion', 'Indian Politics', 'Data Science', etc.
        Description (if available): {description}

        Return only a comma-separated list of subjects. Do not include any explanations or additional text.
        """

        model = genai.GenerativeModel("gemini-2.5-flash")  # Fixed model name
        response = model.generate_content(prompt)
        subjects_raw = response.text.strip()

        # Clean up the response - remove any markdown, numbers, etc.
        subjects_raw = re.sub(r'[\d\.\-•]', '', subjects_raw)  # Remove numbers and bullets
        subjects_raw = re.sub(r'\*+', '', subjects_raw)  # Remove asterisks
        
        # Convert to list and clean each subject
        subjects = []
        for s in subjects_raw.split(","):
            s_clean = s.strip()
            if s_clean and len(s_clean) > 2:  # Filter out very short strings
                subjects.append(s_clean)
        
        # Limit to 5 subjects and ensure we have at least one
        subjects = subjects[:5]
        if not subjects:
            subjects = ["General"]
            
        print(f"📚 Gemini-generated subjects: {subjects}")
        return subjects

    except Exception as e:
        print(f"⚠️ Gemini subject generation failed: {e}")
        # Fallback subjects based on common keywords in title
        fallback_subjects = []
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['programming', 'code', 'software', 'computer']):
            fallback_subjects.append("Computer Science")
        if any(word in title_lower for word in ['data', 'analytics', 'statistics']):
            fallback_subjects.append("Data Science")
        if any(word in title_lower for word in ['business', 'management', 'marketing']):
            fallback_subjects.append("Business")
        if any(word in title_lower for word in ['history', 'historical']):
            fallback_subjects.append("History")
        if any(word in title_lower for word in ['fiction', 'novel', 'story']):
            fallback_subjects.append("Fiction")
            
        return fallback_subjects if fallback_subjects else ["General"]
