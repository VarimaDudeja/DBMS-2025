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

# Basic text extraction function
def extract_text(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply different preprocessing techniques
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Try multiple OCR configurations
        configs = [
            '--psm 6',  # Uniform block of text
            '--psm 4',  # Single column of text  
            '--psm 3',  # Fully automatic page segmentation
            '--psm 11'  # Sparse text
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

# Extract comprehensive data from both images
def extract_all_possible_data(image1_path, image2_path):
    """
    Extract all possible book data from both images
    Returns dict with title, authors, publisher, isbn, publication_date
    """
    print("🔍 Extracting all possible data from images...")
    
    # Extract from both images
    text1 = extract_text(image1_path)
    text2 = extract_text(image2_path)
    combined_text = text1 + "\n" + text2
    
    print("Combined extracted text from both images:")
    print("=" * 50)
    print(combined_text)
    print("=" * 50)
    
    extracted_data = {
        "title": "",
        "authors": [],
        "publisher": "",
        "isbn": "",
        "publication_date": ""
    }
    
    lines = [line.strip() for line in combined_text.split('\n') if line.strip()]
    
    # Extract ISBN (highest priority - most reliable from images)
    isbn_patterns = [
        r'ISBN[-:\s]*([\d\-Xx]{10,17})',
        r'ISBN\s*[^:\n]*[:]?\s*([\d\-Xx]{10,17})',
        r'([\d\-]{9,17}[Xx]?)'
    ]
    
    for line in lines:
        for pattern in isbn_patterns:
            matches = re.findall(pattern, line, re.IGNORECASE)
            for match in matches:
                isbn_clean = re.sub(r'[^\dXx]', '', match.upper())
                if len(isbn_clean) in [10, 13] and not extracted_data["isbn"]:
                    extracted_data["isbn"] = isbn_clean
                    print(f"✅ Extracted ISBN from image: {isbn_clean}")
                    break
    
    # Extract Title
    for i, line in enumerate(lines[:8]):  # Check first 8 lines
        line_clean = re.sub(r'[^\w\s\-\',\.!?]', '', line).strip()
        if (len(line_clean) >= 5 and 
            line_clean[0].isupper() and
            2 <= len(line_clean.split()) <= 8 and
            not any(keyword in line_clean.lower() for keyword in [
                'by', 'author', 'publisher', 'published', 'copyright', 
                'edition', 'isbn', 'sale', 'cover', 'only'
            ])):
            extracted_data["title"] = line_clean
            print(f"✅ Extracted title from image: '{line_clean}'")
            break
    
    # Extract Authors
    for line in lines:
        line_lower = line.lower()
        if 'by ' in line_lower and len(line) < 100:
            author_part = line.split('by ')[-1].strip()
            # Clean author name
            author_clean = re.sub(r'[^\w\s\-\',\.]', '', author_part).strip()
            if (author_clean and len(author_clean.split()) <= 4 and
                not any(pub_word in author_clean.lower() for pub_word in ['inc', 'ltd', 'media', 'press'])):
                extracted_data["authors"].append(author_clean)
                print(f"✅ Extracted author from image: '{author_clean}'")
    
    # Extract Publisher
    for line in lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in ['published by', 'publisher:']):
            if 'published by' in line_lower:
                pub_part = line.split('published by')[-1].strip()
            else:
                pub_part = line.split('publisher:')[-1].strip()
            
            pub_clean = re.sub(r'[,\s]*(?:all rights|reserved|copyright).*$', '', pub_part, flags=re.IGNORECASE)
            if pub_clean and len(pub_clean) > 3:
                extracted_data["publisher"] = pub_clean
                print(f"✅ Extracted publisher from image: '{pub_clean}'")
                break
    
    # Extract Publication Date
    date_patterns = [
        r'(?:published|copyright|©)\s*(?:in\s*)?(\d{4})',
        r'(\d{4})\s*(?:edition|printing)',
        r'first published\s*(\d{4})'
    ]
    
    for line in lines:
        for pattern in date_patterns:
            date_match = re.search(pattern, line.lower())
            if date_match and not extracted_data["publication_date"]:
                extracted_data["publication_date"] = date_match.group(1)
                print(f"✅ Extracted publication date from image: {date_match.group(1)}")
                break
    
    print("📋 Image extraction summary:")
    print(f"   Title: {extracted_data['title']}")
    print(f"   Authors: {extracted_data['authors']}")
    print(f"   Publisher: {extracted_data['publisher']}")
    print(f"   ISBN: {extracted_data['isbn']}")
    print(f"   Publication Date: {extracted_data['publication_date']}")
    
    return extracted_data

def parse_google_books_response(book_item):
    """Parse Google Books API response into standardized format"""
    book_info = book_item["volumeInfo"]
    
    # Extract industry identifiers
    isbn_10 = ""
    isbn_13 = ""
    for identifier in book_info.get("industryIdentifiers", []):
        if identifier["type"] == "ISBN_10":
            isbn_10 = identifier["identifier"]
        elif identifier["type"] == "ISBN_13":
            isbn_13 = identifier["identifier"]
    
    return {
        "title": book_info.get("title", ""),
        "authors": book_info.get("authors", []),
        "publisher": book_info.get("publisher", ""),
        "publishedDate": book_info.get("publishedDate", ""),
        "description": book_info.get("description", ""),
        "pageCount": book_info.get("pageCount", ""),
        "categories": book_info.get("categories", []),
        "isbn_10": isbn_10,
        "isbn_13": isbn_13,
        "language": book_info.get("language", ""),
        "source": "google_books"
    }

def fetch_book_data_by_isbn(isbn):
    try:
        clean_isbn = re.sub(r'[^\dXx]', '', isbn).upper()
        if not clean_isbn:
            return None
            
        url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{clean_isbn}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "items" in data and len(data["items"]) > 0:
                return parse_google_books_response(data["items"][0])
        return None
            
    except Exception as e:
        print(f"ISBN API error: {e}")
        return None

def fetch_book_data_by_title(title):
    try:
        clean_title = re.sub(r'[^\w\s]', '', title).strip()
        if not clean_title:
            return None
            
        url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{clean_title}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "items" in data and len(data["items"]) > 0:
                return parse_google_books_response(data["items"][0])
        return None
            
    except Exception as e:
        print(f"Title API error: {e}")
        return None

def fetch_book_data_enhanced(extracted_data):
    """
    Try multiple API strategies to get complete book data
    Priority: ISBN lookup (most reliable) > Title lookup
    """
    book_data = None
    
    # STRATEGY 1: ISBN lookup (most reliable)
    if extracted_data["isbn"]:
        print(f"🔍 STRATEGY 1: ISBN lookup for {extracted_data['isbn']}")
        book_data = fetch_book_data_by_isbn(extracted_data["isbn"])
        if book_data:
            print("✅ Success with ISBN lookup")
            print(f"   Authors from ISBN API: {book_data.get('authors', [])}")
            return book_data
    
    # STRATEGY 2: Title lookup
    if extracted_data["title"]:
        print(f"🔍 STRATEGY 2: Title lookup for '{extracted_data['title']}'")
        book_data = fetch_book_data_by_title(extracted_data["title"])
        if book_data:
            print("✅ Success with Title lookup")
            print(f"   Authors from Title API: {book_data.get('authors', [])}")
            return book_data
    
    # STRATEGY 3: Enhanced image extraction with API author lookup
    if extracted_data["title"]:
        print("⚠️ Using image-extracted data with API author enhancement")
        
        # Try to get authors using ISBN if available (most reliable)
        authors_from_api = None
        if extracted_data["isbn"]:
            book_data = fetch_book_data_by_isbn(extracted_data["isbn"])
            if book_data:
                authors_from_api = book_data.get("authors")
        
        # Fallback to title-based author lookup
        if not authors_from_api:
            book_data = fetch_book_data_by_title(extracted_data["title"])
            if book_data:
                authors_from_api = book_data.get("authors")
        
        if authors_from_api:
            print(f"✅ Enhanced authors via API: {authors_from_api}")
            extracted_data["authors"] = authors_from_api
        else:
            print("❌ No authors found via API, using image-extracted authors")
        
        book_data = {
            "title": extracted_data["title"],
            "authors": extracted_data["authors"] or ["Unknown Author"],
            "publisher": extracted_data["publisher"] or "Unknown Publisher",
            "publishedDate": extracted_data["publication_date"] or "",
            "isbn_13": extracted_data["isbn"] or "",
            "source": "image_extraction_with_api_authors"
        }
        return book_data
    
    print("❌ All API strategies failed")
    return None

def extract_subject_with_nlp(title, description=""):
    """
    Use NLP to extract the main subject/topic of the book
    Returns a meaningful subject heading
    """
    try:
        if not nlp:
            print("⚠️ NLP model not available, using title-based subject")
            return title[:100] if title else "General"
        
        # Combine title and description for better context
        text_to_analyze = title
        if description:
            text_to_analyze += ". " + description
        
        print(f"🔍 Analyzing subject for: {text_to_analyze[:100]}...")
        
        # Process text with spaCy
        doc = nlp(text_to_analyze)
        
        # Extract nouns and proper nouns (most likely to be subjects)
        subjects = []
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                # Filter out common words that aren't good subjects
                if token.text.lower() not in ['handbook', 'guide', 'introduction', 'book', 'edition']:
                    subjects.append(token.text)
        
        # Also look for noun phrases
        noun_phrases = [chunk.text for chunk in doc.noun_chunks 
                       if len(chunk.text.split()) <= 3]  # Keep phrases short
        
        # Combine and rank subjects
        all_subjects = subjects + noun_phrases
        
        if all_subjects:
            # Count frequency to find most relevant subject
            from collections import Counter
            subject_counts = Counter(all_subjects)
            most_common_subject = subject_counts.most_common(1)[0][0]
            
            print(f"✅ NLP extracted subject: {most_common_subject}")
            return most_common_subject
        
        # Fallback: use key words from title
        print("⚠️ No specific subject found with NLP, using title keywords")
        return title[:100]
        
    except Exception as e:
        print(f"❌ NLP subject extraction error: {e}")
        return title[:100] if title else "General"

def categorize_subject(subject_text):
    """
    Categorize the subject into broader classification codes
    """
    subject_lower = subject_text.lower()
    
    # Computer Science & Programming
    if any(keyword in subject_lower for keyword in ['python', 'programming', 'code', 'software', 'computer', 'algorithm', 'data structure']):
        return "COM"
    # Data Science & Analytics
    elif any(keyword in subject_lower for keyword in ['data science', 'data analysis', 'machine learning', 'artificial intelligence', 'ai', 'ml', 'statistics']):
        return "DAT"
    # Mathematics
    elif any(keyword in subject_lower for keyword in ['mathematics', 'math', 'calculus', 'algebra', 'geometry']):
        return "MAT"
    # Science
    elif any(keyword in subject_lower for keyword in ['science', 'physics', 'chemistry', 'biology']):
        return "SCI"
    # Business
    elif any(keyword in subject_lower for keyword in ['business', 'management', 'marketing', 'finance', 'economics']):
        return "BUS"
    # Literature
    elif any(keyword in subject_lower for keyword in ['literature', 'fiction', 'novel', 'poetry', 'writing']):
        return "LIT"
    # History
    elif any(keyword in subject_lower for keyword in ['history', 'historical', 'biography']):
        return "HIS"
    # General (default)
    else:
        return "GEN"

# Database insertion function
def insert_into_database(book_data):
    try:
        print("📦 Inserting book via Supabase REST API...")
        
        # Extract data
        title = book_data.get('title', 'Unknown Title').strip()
        authors = book_data.get('authors', [])
        publisher = book_data.get('publisher', 'Unknown Publisher').strip()
        description = book_data.get('description', '')
        
        if not title or title == 'Unknown Title':
            return False, "No valid title available"
        
        # Ensure authors is a list
        if isinstance(authors, str):
            authors = [authors]
        if not authors:
            authors = ['Unknown Author']
        
        # Generate control number
        control_no = f"LBS-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        print(f"📚 Final book data:")
        print(f"   Title: {title}")
        print(f"   Authors: {authors}")
        print(f"   Publisher: {publisher}")
        
        # 1. Insert or get publisher
        print("1. Handling publisher...")
        publisher_response = requests.post(
            f"{SUPABASE_URL}/rest/v1/publishers",
            headers=headers,
            json={"name": publisher},
            params={"on_conflict": "name"}
        )
        
        # 2. Insert authors
        print("2. Handling authors...")
        author_ids = []
        for author in authors:
            if author and author != 'Unknown Author':
                author_response = requests.post(
                    f"{SUPABASE_URL}/rest/v1/authors",
                    headers=headers,
                    json={"name": author, "role": "Author"},
                    params={"on_conflict": "name"}
                )
                if author_response.status_code in [200, 201]:
                    author_data = author_response.json()
                    if isinstance(author_data, list) and len(author_data) > 0:
                        author_ids.append(author_data[0]['author_id'])
        
        # 3. Insert subject USING NLP (NEW IMPROVEMENT)
        print("3. Handling subject with NLP...")
        subject_heading = extract_subject_with_nlp(title, description)
        classification_code = categorize_subject(subject_heading)
        
        print(f"   Subject: {subject_heading}")
        print(f"   Classification: {classification_code}")
        
        subject_response = requests.post(
            f"{SUPABASE_URL}/rest/v1/subjects",
            headers=headers,
            json={
                "subject_heading": subject_heading,
                "classification_code": classification_code
            },
            params={"on_conflict": "subject_heading"}
        )
        
        # 4. Get publisher_id
        print("4. Getting publisher ID...")
        publishers_response = requests.get(
            f"{SUPABASE_URL}/rest/v1/publishers?name=eq.{publisher}",
            headers=headers
        )
        
        publisher_id = 1  # Default fallback
        if publishers_response.status_code == 200:
            publishers_data = publishers_response.json()
            if publishers_data and len(publishers_data) > 0:
                publisher_id = publishers_data[0]['publisher_id']
        
        # 5. Insert MARC record
        print("5. Inserting MARC record...")
        marc_response = requests.post(
            f"{SUPABASE_URL}/rest/v1/marc_records",
            headers=headers,
            json={
                "control_no": control_no,
                "record_type": "Book",
                "publisher_id": publisher_id
            }
        )
        
        if marc_response.status_code not in [200, 201]:
            return False, f"Failed to create MARC record: {marc_response.text}"
        
        marc_data = marc_response.json()
        if isinstance(marc_data, list) and len(marc_data) > 0:
            record_id = marc_data[0]['record_id']
            
            # 6. Link authors to record
            print("6. Linking authors...")
            for author_id in author_ids:
                requests.post(
                    f"{SUPABASE_URL}/rest/v1/record_authors",
                    headers=headers,
                    json={
                        "record_id": record_id,
                        "author_id": author_id
                    }
                )
            
            # 7. Link subject to record
            print("7. Linking subject...")
            subjects_response = requests.get(
                f"{SUPABASE_URL}/rest/v1/subjects?subject_heading=eq.{subject_heading}",
                headers=headers
            )
            if subjects_response.status_code == 200:
                subjects_data = subjects_response.json()
                if subjects_data and len(subjects_data) > 0:
                    subject_id = subjects_data[0]['subject_id']
                    requests.post(
                        f"{SUPABASE_URL}/rest/v1/record_subjects",
                        headers=headers,
                        json={
                            "record_id": record_id,
                            "subject_id": subject_id
                        }
                    )
            
            # 8. Insert MARC fields
            print("8. Inserting MARC fields...")
            # Title field (245)
            requests.post(
                f"{SUPABASE_URL}/rest/v1/marc_fields",
                headers=headers,
                json={
                    "record_id": record_id,
                    "tag": "245",
                    "indicators": "00",
                    "field_value": title
                }
            )
            
            # Add description field (520) if available
            if description:
                requests.post(
                    f"{SUPABASE_URL}/rest/v1/marc_fields",
                    headers=headers,
                    json={
                        "record_id": record_id,
                        "tag": "520",
                        "indicators": "00", 
                        "field_value": description[:500]  # Limit length
                    }
                )
            
            print("✅ Book inserted successfully via REST API!")
            return True, f"Book '{title}' successfully added with Control No: {control_no}"
        
        return False, "Failed to get record ID from response"
            
    except Exception as e:
        print(f"❌ API call failed: {str(e)}")
        return False, f"API call failed: {str(e)}"

# Main processing function
def process_two_images_and_save(image1_path, image2_path):
    try:
        print("🚀 Starting enhanced book processing...")
        print("=" * 60)
        
        # STEP 1: Extract whatever we can from images
        extracted_data = extract_all_possible_data(image1_path, image2_path)
        
        print("\n" + "=" * 60)
        print("🌐 ATTEMPTING API LOOKUP WITH EXTRACTED DATA...")
        
        # STEP 2: Use API as primary source with multiple strategies
        api_book_data = fetch_book_data_enhanced(extracted_data)
        
        # STEP 3: Combine API data with extracted data (API takes priority)
        final_book_data = {}
        
        if api_book_data:
            print("✅ Using API data as primary source")
            final_book_data = api_book_data
            
            # Only use image data if API doesn't have it
            if not final_book_data.get("isbn_10") and not final_book_data.get("isbn_13"):
                if extracted_data["isbn"]:
                    final_book_data["isbn_13"] = extracted_data["isbn"]
            
            # Update source to indicate combination
            final_book_data["source"] = "api_primary_with_image_fallback"
        else:
            print("⚠️ API lookup failed, using image data as fallback")
            final_book_data = {
                "title": extracted_data["title"] or "Unknown Title",
                "authors": extracted_data["authors"] or ["Unknown Author"],
                "publisher": extracted_data["publisher"] or "Unknown Publisher",
                "publishedDate": extracted_data["publication_date"] or "",
                "isbn_13": extracted_data["isbn"] or "",
                "source": "image_extraction_fallback"
            }
        
        print("\n" + "=" * 60)
        print("🎯 FINAL COMBINED BOOK DATA:")
        print(f"   Title: {final_book_data['title']}")
        print(f"   Authors: {final_book_data['authors']}")
        print(f"   Publisher: {final_book_data['publisher']}")
        print(f"   Published Date: {final_book_data.get('publishedDate', '')}")
        print(f"   ISBN: {final_book_data.get('isbn_13', final_book_data.get('isbn_10', ''))}")
        print(f"   Source: {final_book_data.get('source', 'unknown')}")
        
        # Validate minimum data
        if final_book_data["title"] == "Unknown Title":
            return False, "Could not extract sufficient book information from images or API.", final_book_data
        
        # STEP 4: Insert into database
        print("\n💾 SAVING TO DATABASE...")
        success, message = insert_into_database(final_book_data)
        
        return success, message, final_book_data
        
    except Exception as e:
        error_msg = f"Error processing images: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return False, error_msg, None