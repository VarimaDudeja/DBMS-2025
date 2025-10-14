-- Check if publisher exists
SELECT * FROM publishers WHERE name = 'publisher_name';

-- Check if author exists  
SELECT * FROM authors WHERE name = 'author_name';

-- Check if subject exists
SELECT * FROM subjects WHERE subject_heading = 'subject_heading';

-- Check if MARC record exists
SELECT * FROM marc_records WHERE control_no = 'control_number';

-- Check if record-author link exists
SELECT * FROM record_authors WHERE record_id = 'record_id' AND author_id = 'author_id';

-- Check if record-subject link exists
SELECT * FROM record_subjects WHERE record_id = 'record_id' AND subject_id = 'subject_id';

-- Get all publishers (fallback query)
SELECT * FROM publishers;

-- Insert new publisher
INSERT INTO publishers (name) VALUES ('publisher_name');

-- Insert new author
INSERT INTO authors (name, role) VALUES ('author_name', 'Author');

-- Insert new subject  
INSERT INTO subjects (subject_heading, classification_code) VALUES ('subject_heading', 'GEN');

-- Insert new MARC record
INSERT INTO marc_records (control_no, record_type, publisher_id) VALUES ('control_number', 'Book', publisher_id);

-- Link record with author
INSERT INTO record_authors (record_id, author_id) VALUES (record_id, author_id);

-- Link record with subject
INSERT INTO record_subjects (record_id, subject_id) VALUES (record_id, subject_id);

-- Insert MARC fields
INSERT INTO marc_fields (record_id, tag, indicators, field_value) VALUES 
(record_id, '245', '00', 'title'),
(record_id, '100', '00', 'author'),
(record_id, '020', '00', 'ISBN number'),
(record_id, '260', '00', 'publication_date'),
(record_id, '250', '00', 'edition'),
(record_id, '264', '00', 'publisher');


-- Publisher upsert (insert or ignore if exists)
INSERT INTO publishers (name) VALUES ('publisher_name') 
ON CONFLICT (name) DO NOTHING;

-- Author upsert  
INSERT INTO authors (name, role) VALUES ('author_name', 'Author')
ON CONFLICT (name) DO NOTHING;

-- Subject upsert
INSERT INTO subjects (subject_heading, classification_code) VALUES ('subject_heading', 'GEN')
ON CONFLICT (subject_heading) DO NOTHING;
