-- Insert publisher (ignore if already exists)
import json

# Assuming your extracted dict is stored in variable `data`
record = {
    "title": data.get("title"),
    "author": data.get("author"),
    "edition": data.get("edition"),
    "isbn_international": data.get("ISBN", {}).get("international"),
    "isbn_indian": data.get("ISBN", {}).get("indian_reprint"),
    "publisher_original": data.get("publisher", {}).get("original"),
    "publisher_reprint": data.get("publisher", {}).get("reprint"),
    "published_date": data.get("published_date"),
    "contributors": json.dumps(data.get("contributors"))  # store as JSON string
}

# Then your Supabase insert (example)
supabase.table("books").insert(record).execute()
 
INSERT INTO publishers (name)
VALUES ('{publisher_name}')
ON CONFLICT (name) DO NOTHING;

-- Insert author (ignore if already exists)
INSERT INTO authors (name, role)
VALUES ('{author_name}', 'Author')
ON CONFLICT (name) DO NOTHING;

-- Insert subject (ignore if already exists)
INSERT INTO subjects (subject_heading, classification_code)
VALUES ('{subject}', '{classification_code}')
ON CONFLICT (subject_heading) DO NOTHING;

-- Insert MARC record (main book info)
INSERT INTO marc_records (control_no, record_type, publisher_id)
VALUES (
  '{control_no}',
  'Book',
  (SELECT publisher_id FROM publishers WHERE name='{publisher_name}')
);

-- Link record with author
INSERT INTO record_authors (record_id, author_id)
VALUES (
  (SELECT record_id FROM marc_records WHERE control_no='{control_no}'),
  (SELECT author_id FROM authors WHERE name='{author_name}')
);

-- Link record with subject
INSERT INTO record_subjects (record_id, subject_id)
VALUES (
  (SELECT record_id FROM marc_records WHERE control_no='{control_no}'),
  (SELECT subject_id FROM subjects WHERE subject_heading='{subject}')
);
