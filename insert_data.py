# insert_data.py
import os
import re
from datetime import datetime
from supabase import create_client, Client

# ---------------------------------------------------------
#  CONFIG: prefer environment variables for safety
# ---------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://lvopadqxajbvcwxzucta.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx2b3BhZHF4YWpidmN3eHp1Y3RhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk0ODk2OTMsImV4cCI6MjA3NTA2NTY5M30.Cn033yAamWufnwKvs3I77cj4LeJR5q-0uH8YE6WmnlA")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
try:
    result = supabase.table("publishers").select("*").limit(1).execute()
    print("Connection successful!")
    print(result)
except Exception as e:
    print(" Connection failed:", e)
def create_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Set SUPABASE_URL and SUPABASE_KEY environment variables.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------
#  CLEANING / EXTRACTION HELPERS
# ---------------------------------------------------------
def _collapse_whitespace(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def clean_text(value, max_len=255, allow_punct=False):
    """Basic cleaning of short text fields (names, publisher names)."""
    if not value:
        return None
    v = str(value)
    v = _collapse_whitespace(v)
    if not allow_punct:
        # allow basic punctuation but remove weird chars
        v = re.sub(r'[^A-Za-z0-9&.,:\'"\-\s()]', '', v)
    else:
        v = re.sub(r'[^\x00-\x7F]', '', v)  # remove non-ascii weird chars
    v = v.strip()
    return v[:max_len] if max_len else v


def extract_isbn(raw):
    """Try to extract a 10 or 13 digit ISBN (strip hyphens/spaces)."""
    if not raw:
        return None
    s = str(raw)
    # look for groups that contain digits, hyphens, spaces and X
    candidates = re.findall(r'[\dXx\-\s]{10,20}', s)
    for cand in candidates:
        cleaned = re.sub(r'[^0-9Xx]', '', cand)
        if len(cleaned) in (10, 13):
            return cleaned.upper()
    # fallback: any single run of 10 or 13 digits
    m = re.search(r'\b(\d{10}|\d{13})\b', s)
    if m:
        return m.group(0)
    return None


def extract_year(raw):
    """Return the first plausible 4-digit year (1500-2099)."""
    if not raw:
        return None
    s = str(raw)
    m = re.search(r'\b(1[5-9]\d{2}|20[0-9]{2})\b', s)
    if m:
        return m.group(0)
    return None


def clean_publisher(raw_text):
    """Return the best candidate publisher name from noisy text."""
    if not raw_text:
        return None
    s = str(raw_text)
    # split on commas / newlines and prefer segments with publisher-like words
    parts = re.split(r'[,|\n]+', s)
    # candidate scoring
    keywords = ['media', 'press', 'publisher', 'publications', 'publishing', 'house', 'edition', 'inc', "ltd"]
    for p in parts:
        if any(k in p.lower() for k in keywords):
            return clean_text(p)
    # fallback: choose first part that looks short & not copyright noise
    for p in parts:
        p_clean = clean_text(p)
        if p_clean and len(p_clean.split()) <= 6 and not re.search(r'copyright|protected|reprint|all rights', p_clean, re.I):
            return p_clean
    # last resort: truncated full cleaned string
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
        # remove obvious copyright/footer lines or lines with too many words
        if any(w in c.lower() for w in invalid_words):
            continue
        if any(char.isdigit() for char in c):
            continue
        # limit to 1-4 words (typical person name)
        words = c.split()
        if not (1 <= len(words) <= 4):
            continue
        # rule out tiny tokens and single-letter tokens
        if any(len(w) <= 1 for w in words):
            continue
        cleaned.append(c)

    # dedupe preserving order
    seen = set()
    result = []
    for name in cleaned:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            result.append(name)
    return result


def clean_title(raw_title, raw_publisher=None, raw_authors=None):
    """Try to return a clean title. Use provided title if valid, else try fallback scanning."""
    # 1) if provided and not just 'Unknown'
    if raw_title and isinstance(raw_title, str) and raw_title.strip() and raw_title.lower() != 'unknown':
        return clean_text(raw_title, max_len=1000, allow_punct=True)

    # 2) check publisher blobs for a likely title (some scans put title next to publisher)
    if raw_publisher:
        parts = re.split(r'[,|\n]+', str(raw_publisher))
        for p in parts:
            p_clean = clean_text(p, max_len=1000, allow_punct=True)
            if p_clean and 2 <= len(p_clean.split()) <= 10 and not re.search(r'publisher|media|press|inc|copyright|reprint', p_clean, re.I):
                return p_clean

    # 3) try authors (sometimes title is before an author string)
    if raw_authors and isinstance(raw_authors, str):
        # if authors blob contains a line before names that looks like a title, try it
        parts = raw_authors.split('\n')
        if len(parts) >= 2:
            candidate = clean_text(parts[0], max_len=1000, allow_punct=True)
            if candidate and 2 <= len(candidate.split()) <= 10:
                return candidate

    # fallback
    return "Unknown"


# ---------------------------------------------------------
#  DB INSERT HELPERS (Supabase)
# ---------------------------------------------------------
def get_or_create_publisher(supabase: Client, publisher_name):
    publisher_name = clean_publisher(publisher_name)
    if not publisher_name:
        return None
    # safe truncate
    publisher_name = publisher_name[:255]
    existing = supabase.table("publishers").select("publisher_id").eq("name", publisher_name).limit(1).execute()
    if existing.data and len(existing.data) > 0:
        return existing.data[0]["publisher_id"]
    res = supabase.table("publishers").insert({"name": publisher_name}).execute()
    if not res.data:
        raise RuntimeError(f"Failed to insert publisher: {publisher_name}")
    return res.data[0]["publisher_id"]


def get_or_create_author(supabase: Client, author_name):
    author_name = clean_text(author_name)
    if not author_name:
        return None
    author_name = author_name[:255]
    existing = supabase.table("authors").select("author_id").eq("name", author_name).limit(1).execute()
    if existing.data and len(existing.data) > 0:
        return existing.data[0]["author_id"]
    res = supabase.table("authors").insert({"name": author_name, "role": "Author"}).execute()
    if not res.data:
        raise RuntimeError(f"Failed to insert author: {author_name}")
    return res.data[0]["author_id"]


def get_or_create_record(supabase: Client, control_no, record_type, publisher_id):
    """If control_no (ISBN) exists return record_id, else create new marc_records row."""
    control_clean = clean_text(control_no) if control_no else None
    if control_clean:
        existing = supabase.table("marc_records").select("record_id").eq("control_no", control_clean).limit(1).execute()
        if existing.data and len(existing.data) > 0:
            return existing.data[0]["record_id"]

    payload = {
        "control_no": control_clean or "Unknown",
        "record_type": record_type or "Book",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "publisher_id": publisher_id
    }
    res = supabase.table("marc_records").insert(payload).execute()
    if not res.data:
        raise RuntimeError("Failed to insert marc_records")
    return res.data[0]["record_id"]


def link_record_author(supabase: Client, record_id, author_id):
    if not record_id or not author_id:
        return
    existing = supabase.table("record_authors").select("*").eq("record_id", record_id).eq("author_id", author_id).limit(1).execute()
    if existing.data and len(existing.data) > 0:
        return
    supabase.table("record_authors").insert({"record_id": record_id, "author_id": author_id}).execute()


def insert_marc_field(supabase: Client, record_id, tag, indicators, field_value):
    """Insert a marc_fields row and return field_id."""
    payload = {
        "record_id": record_id,
        "tag": str(tag),
        "indicators": indicators or "",
        "field_value": field_value
    }
    res = supabase.table("marc_fields").insert(payload).execute()
    if not res.data:
        return None
    return res.data[0]["field_id"]


def insert_marc_subfield(supabase: Client, field_id, code, value):
    if not field_id or not value:
        return
    payload = {
        "field_id": field_id,
        "code": code,
        "value": value
    }
    supabase.table("marc_subfields").insert(payload).execute()


# ---------------------------------------------------------
#  MAIN: store_extracted_data
# ---------------------------------------------------------
def store_extracted_data(supabase: Client, book_data: dict):

    """
    book_data expected keys:
      - title (str or list)
      - authors (str or list)
      - publisher (str)
      - publishedDate (str)
      - ISBN (str)
    """
    print("Cleaning extracted data...")

    # Clean / extract
   # --- Normalize nested structured fields ---
    isbn_info = book_data.get("ISBN", {})
    publisher_info = book_data.get("publisher", {})

    raw_title = book_data.get("title")
    raw_authors = book_data.get("author") or book_data.get("authors")
    raw_publisher = publisher_info.get("original") or publisher_info.get("reprint") or book_data.get("publisher")
    raw_isbn = (
        isbn_info.get("international")
        or isbn_info.get("indian_reprint")
        or book_data.get("ISBN")
        or extract_isbn(book_data.get("isbn") or book_data.get("control_no") or "")
    )
    raw_date = (
        book_data.get("published_date")
        or book_data.get("publishedDate")
        or extract_year(book_data.get("publishedDate") or "")
    )


    title = clean_title(raw_title, raw_publisher, raw_authors)
    authors = clean_authors(raw_authors)
    publisher = clean_publisher(raw_publisher)
    isbn = extract_isbn(raw_isbn) or extract_isbn(publisher) or extract_isbn(raw_title) or None
    pub_year = extract_year(raw_date) or extract_year(raw_publisher) or None

    print(f"Title: {title}")
    print(f"Authors: {authors}")
    print(f"Publisher: {publisher}")
    print(f"ISBN: {isbn}")
    print(f"PublishedYear: {pub_year}")

    # Insert publisher
    publisher_id = get_or_create_publisher(supabase, publisher) if publisher else None

    # Insert record (use ISBN as control_no if available)
    record_id = get_or_create_record(supabase, isbn, "Book", publisher_id)
    print(" store_extracted_data() called for record:", record_id)
    # ---------- Insert MARC fields ----------
    # Title -> tag 245 (common MARC)
    title_field_id = insert_marc_field(supabase, record_id, "245", "1 ", title)
    if title_field_id:
        insert_marc_subfield(supabase, title_field_id, "a", title)

    # Main author -> 100 (first author)
    if authors:
        main_author = authors[0]
        author_id = get_or_create_author(supabase, main_author)
        link_record_author(supabase, record_id, author_id)

        # create MARC 100
        a_field_id = insert_marc_field(supabase, record_id, "100", "1 ", main_author)
        if a_field_id:
            insert_marc_subfield(supabase, a_field_id, "a", main_author)

    # Additional authors -> 700
    if len(authors) > 1:
        for a in authors[1:]:
            a_id = get_or_create_author(supabase, a)
            link_record_author(supabase, record_id, a_id)
            f_id = insert_marc_field(supabase, record_id, "700", "1 ", a)
            if f_id:
                insert_marc_subfield(supabase, f_id, "a", a)

    # Publisher/date -> MARC 260 (older) or 264
    pub_value = publisher if publisher else ""
    if pub_year:
        pub_value = f"{pub_value} {pub_year}".strip()
    if pub_value:
        p_field_id = insert_marc_field(supabase, record_id, "260", "  ", pub_value)
        if p_field_id:
            insert_marc_subfield(supabase, p_field_id, "b", publisher or "")
            if pub_year:
                insert_marc_subfield(supabase, p_field_id, "c", pub_year)

    print("Data written to Supabase (publisher, record, authors, marc_fields/subfields).")
    return {
        "record_id": record_id,
        "publisher_id": publisher_id,
        "authors_inserted": authors,
        "title": title,
        "isbn": isbn
    }
