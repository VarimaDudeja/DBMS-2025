-- Authors Table
CREATE TABLE authors (
    author_id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    role VARCHAR(100)
);

-- Publishers Table
CREATE TABLE publishers (
    publisher_id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL
);

-- Subjects Table
CREATE TABLE subjects (
    subject_id SERIAL PRIMARY KEY,
    subject_heading VARCHAR(255) UNIQUE,
    classification_code VARCHAR(100)
);

-- MARC Records Table
CREATE TABLE marc_records (
    record_id SERIAL PRIMARY KEY,
    control_no VARCHAR(50),
    record_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    publisher_id INT REFERENCES publishers(publisher_id)
);

-- MARC Fields Table
CREATE TABLE marc_fields (
    field_id SERIAL PRIMARY KEY,
    record_id INT REFERENCES marc_records(record_id) ON DELETE CASCADE,
    tag VARCHAR(10),
    indicators VARCHAR(10),
    field_value TEXT
);

-- MARC Subfields Table
CREATE TABLE marc_subfields (
    subfield_id SERIAL PRIMARY KEY,
    field_id INT REFERENCES marc_fields(field_id) ON DELETE CASCADE,
    code VARCHAR(10),
    value TEXT
);

-- Relationship: marc_records ↔ authors (Many-to-Many)
CREATE TABLE record_authors (
    record_id INT REFERENCES marc_records(record_id) ON DELETE CASCADE,
    author_id INT REFERENCES authors(author_id) ON DELETE CASCADE,
    PRIMARY KEY (record_id, author_id)
);

-- Relationship: marc_records ↔ subjects (Many-to-Many)
CREATE TABLE record_subjects (
    record_id INT REFERENCES marc_records(record_id) ON DELETE CASCADE,
    subject_id INT REFERENCES subjects(subject_id) ON DELETE CASCADE,
    PRIMARY KEY (record_id, subject_id)
);
ALTER TABLE publishers
ADD CONSTRAINT unique_publisher_name UNIQUE (name);

ALTER TABLE authors
ADD CONSTRAINT unique_author_name UNIQUE (name);

ALTER TABLE subjects
ADD CONSTRAINT unique_subject_heading UNIQUE (subject_heading);
