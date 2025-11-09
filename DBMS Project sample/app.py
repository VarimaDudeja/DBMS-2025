from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import io
import pandas as pd
from werkzeug.utils import secure_filename
from ExtractingInfo import process_two_images_and_save, check_book_exists, update_book_count, extract_isbn_from_text, extract_text, extract_isbn_for_check
from datetime import datetime, timedelta, timezone
import requests  # Added missing import
import pytz

app = Flask(__name__)
CORS(app)

SUPABASE_URL = "https://ryxijguahtthyxxfpkek.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ5eGlqZ3VhaHR0aHl4eGZwa2VrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjAwODUyMTksImV4cCI6MjA3NTY2MTIxOX0.9Ogoz11V59q69ZUVemUY0u59459CGBLtrEg1GERrOYQ"

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/process-book', methods=['POST'])
def process_book():
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'success': False, 'message': 'Both images are required'}), 400

        image1 = request.files['image1']
        image2 = request.files['image2']

        if image1.filename == '' or image2.filename == '':
            return jsonify({'success': False, 'message': 'No selected files'}), 400

        if not (allowed_file(image1.filename) and allowed_file(image2.filename)):
            return jsonify({'success': False, 'message': 'Invalid file type'}), 400

        filename1 = secure_filename(image1.filename)
        filename2 = secure_filename(image2.filename)
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        image1.save(filepath1)
        image2.save(filepath2)
        print(f"✅ Saved files: {filename1}, {filename2}")

        # Step 1: Extract ISBN before doing any insertion
        # ============================================
        # Step 1: Extract ISBN and Check for Duplicates
        isbn = extract_isbn_for_check(filepath1, filepath2)
        print(f"🔎 Extracted ISBN for duplicate check: {isbn}")

        if isbn:
            existing_book = check_book_exists(isbn)
            if existing_book:
                print(f"⚠️ DUPLICATE FOUND — Skipping insertion for control_no: {existing_book['control_no']}")

                # Clean up temporary files
                try:
                    os.remove(filepath1)
                    os.remove(filepath2)
                except OSError as e:
                    print(f"Cleanup warning: {e}")

                # ✅ Return consistent duplicate response
                response = jsonify({
                    'success': False,
                    'book_exists': True,
                    'existing_book': existing_book,
                    'message': 'Book already exists. Do you want to increase its count?'
                })
                print("🚫 Returning early to frontend — no insertion done.")
                return response  # STOP HERE
        else:
            print("⚠️ No ISBN detected; continuing with insertion as new record.")

        # ============================================
        # STEP 2: Proceed with Insertion Only If New
        # ============================================
        print("✅ No duplicate detected — inserting new record...")
        success, message, book_data = process_two_images_and_save(filepath1, filepath2)
        print(f"Processing result: {success}, {message}")


        # Cleanup uploaded files
        try:
            os.remove(filepath1)
            os.remove(filepath2)
        except OSError as e:
            print(f"Cleanup warning: {e}")

        print(f"📘 Final status: success={success}, book_exists=False")

        return jsonify({
            'success': success,
            'message': message,
            'book_data': book_data,
            'book_exists': False
        }), 200

    except Exception as e:
        print(f"❌ Error in process-book: {str(e)}")
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500


@app.route('/api/update-book-count', methods=['POST'])
def update_book_count_route():
    try:
        data = request.json
        control_no = data.get('control_no')
        action = data.get('action')  # 'increment' or 'cancel'
        
        if not control_no:
            return jsonify({'success': False, 'message': 'Control number is required'}), 400
        
        if action == 'increment':
            success, message = update_book_count(control_no)
            if success:
                return jsonify({'success': True, 'message': message})
            else:
                return jsonify({'success': False, 'message': message}), 400
        else:
            return jsonify({'success': True, 'message': 'No changes made to book count'})
            
    except Exception as e:
        print(f"Error in update-book-count: {str(e)}")
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

def extract_isbn_for_check(image1_path, image2_path):
    """Extract ISBN quickly for duplicate checking"""
    try:
        # NOTE: Removed redundant imports as they are now at the top of the file
        # Try both images for ISBN
        for image_path in [image1_path, image2_path]:
            text = extract_text(image_path) # Now imported
            isbn = extract_isbn_from_text(text) # Now imported
            if isbn:
                return isbn
        return None
    except Exception as e:
        print(f"Error extracting ISBN for check: {e}")
        return None
    
@app.route('/api/search-books', methods=['GET'])
def search_books():
    try:
        query = request.args.get('q', '').strip()
        
        # Use REST API to search
        if query:
            # Search in titles from marc_fields
            search_response = requests.get(
                f"{SUPABASE_URL}/rest/v1/marc_fields?tag=eq.245&field_value=ilike.%25{query}%25",
                headers=headers
            )
        else:
            # Get all titles
            search_response = requests.get(
                f"{SUPABASE_URL}/rest/v1/marc_fields?tag=eq.245",
                headers=headers
            )
        
        if search_response.status_code != 200:
            return jsonify({'success': False, 'message': f'Search API error: {search_response.status_code}'})
        
        books_data = search_response.json()
        books = []
        
        for book_field in books_data:
            record_id = book_field['record_id']
            title = book_field['field_value']
            
            # Get record details
            record_response = requests.get(
                f"{SUPABASE_URL}/rest/v1/marc_records?record_id=eq.{record_id}&select=*,publishers(name)",
                headers=headers
            )
            
            if record_response.status_code == 200:
                record_data = record_response.json()
                if record_data:
                    record = record_data[0]
                    
                    # Get authors
                    authors_response = requests.get(
                        f"{SUPABASE_URL}/rest/v1/record_authors?record_id=eq.{record_id}&select=authors(name)",
                        headers=headers
                    )
                    authors = []
                    if authors_response.status_code == 200:
                        authors_data = authors_response.json()
                        authors = [author['authors']['name'] for author in authors_data if 'authors' in author and author['authors']]
                    
                    # Get subjects
                    subjects_response = requests.get(
                        f"{SUPABASE_URL}/rest/v1/record_subjects?record_id=eq.{record_id}&select=subjects(subject_heading)",
                        headers=headers
                    )
                    subjects = []
                    if subjects_response.status_code == 200:
                        subjects_data = subjects_response.json()
                        subjects = [subject['subjects']['subject_heading'] for subject in subjects_data if 'subjects' in subject and subject['subjects']]
                    
                    books.append({
                        'record_id': record_id,
                        'control_no': record.get('control_no', 'N/A'),
                        'authors': ', '.join(authors) if authors else 'Unknown Author',
                        'subjects': ', '.join(subjects) if subjects else 'General',
                        'publisher': record.get('publishers', {}).get('name', 'Unknown Publisher') if isinstance(record.get('publishers'), dict) else 'Unknown Publisher',
                        'title': title
                    })
        
        return jsonify({'success': True, 'books': books})
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({'success': False, 'message': f'Search error: {str(e)}'})
@app.route('/api/download-excel', methods=['GET'])
def download_excel():
    """
    Fetch all books (or filtered ones) from Supabase and download as Excel.
    Now includes subjects from the subjects table.
    """
    try:
        # Optional filters from query params
        title_filter = request.args.get('title', '').strip()
        author_filter = request.args.get('author', '').strip()
        subject_filter = request.args.get('subject', '').strip()

        # Step 1: Get all book titles (tag 245) with record info
        books_url = f"{SUPABASE_URL}/rest/v1/marc_fields?tag=eq.245&select=field_value,record_id,marc_records(control_no,created_at,publishers(name))"
        if title_filter:
            books_url += f"&field_value=ilike.%25{title_filter}%25"

        books_response = requests.get(books_url, headers=headers)
        books_response.raise_for_status()
        books_data = books_response.json()

        records = []

        for book in books_data:
            title = book['field_value']
            record_id = book['record_id']
            record_info = book.get('marc_records', {})
            control_no = record_info.get('control_no', '')
            created_at = record_info.get('created_at', '')
            publisher = record_info.get('publishers', {}).get('name', '')

            # Step 2: Fetch authors (tag 100)
            authors_url = f"{SUPABASE_URL}/rest/v1/marc_fields?record_id=eq.{record_id}&tag=eq.100&select=field_value"
            if author_filter:
                authors_url += f"&field_value=ilike.%25{author_filter}%25"
            authors_response = requests.get(authors_url, headers=headers)
            authors = [a['field_value'] for a in authors_response.json()]

            # Step 3: Fetch subjects from subjects table (via record_subjects) - FIXED
            subjects_url = f"{SUPABASE_URL}/rest/v1/record_subjects?record_id=eq.{record_id}&select=subjects(subject_heading)"
            subjects_response = requests.get(subjects_url, headers=headers)
            
            subjects = []
            if subjects_response.status_code == 200:
                subjects_data = subjects_response.json()
                # Extract subject_heading from the nested structure
                for subject_item in subjects_data:
                    if subject_item.get('subjects'):
                        subject_heading = subject_item['subjects'].get('subject_heading')
                        if subject_heading:
                            subjects.append(subject_heading)
            
            # Apply subject filter if provided
            if subject_filter and not any(subject_filter.lower() in subject.lower() for subject in subjects):
                continue

            # Skip record if author filter didn't match
            if author_filter and not authors:
                continue

            records.append({
                "Title": title,
                "Authors": ", ".join(authors) if authors else "Unknown Author",
                "Publisher": publisher if publisher else "Unknown Publisher",
                "Subjects": ", ".join(subjects) if subjects else "General",
                "Control_No": control_no,
                "Created_At": created_at
            })

        if not records:
            return jsonify({"message": "No matching records found."}), 404

        # Step 4: Convert to Excel
        df = pd.DataFrame(records)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Books')
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Books']
            for idx, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).str.len().max(), len(col)) + 2
                worksheet.set_column(idx, idx, max_len)

        output.seek(0)
        filename = f"libraScan_books_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        return send_file(
            output,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        print(f"Download error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/dashboard-stats', methods=['GET'])
def dashboard_stats():
    try:
        # Get current date in IST
        ist = pytz.timezone('Asia/Kolkata')
        today_ist = datetime.now(ist).date()
        
        # Since timestamps are stored as naive datetime in IST
        # We can query using the local date range
        start_naive = datetime(today_ist.year, today_ist.month, today_ist.day, 0, 0, 0)
        end_naive = start_naive + timedelta(days=1)
        
        # Format for Supabase query
        start_iso = start_naive.isoformat()
        end_iso = end_naive.isoformat()

        print(f"[DEBUG] Querying range: {start_iso} to {end_iso}")

        # Query Supabase
        total_books_res = requests.get(
            f"{SUPABASE_URL}/rest/v1/marc_records?select=record_id",
            headers=headers
        )
        today_books_res = requests.get(
            f"{SUPABASE_URL}/rest/v1/marc_records?created_at=gte.{start_iso}&created_at=lt.{end_iso}&select=record_id",
            headers=headers
        )

        total_count = len(total_books_res.json()) if total_books_res.status_code == 200 else 0
        today_count = len(today_books_res.json()) if today_books_res.status_code == 200 else 0

        return jsonify({
            "success": True,
            "stats": {
                "total_books": total_count,
                "today_records": today_count
            }
        })

    except Exception as e:
        print(f"[ERROR] Dashboard stats error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500
    
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'LibraScan API is running'})


def get_indian_time():
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)

@app.route("/api/get-book/<string:control_no>", methods=["GET"])
def get_book(control_no):
    """Fetch complete book details by control number using MARC tags"""
    try:
        # Step 1: Fetch record info
        record_query = (
            f"{SUPABASE_URL}/rest/v1/marc_records"
            f"?control_no=eq.{control_no}&select=record_id,created_at,updated_at,publishers(name)"
        )
        record_res = requests.get(record_query, headers=headers)
        record_data = record_res.json()

        if not record_data:
            return jsonify({"success": False, "message": "Record not found"}), 404

        record = record_data[0]
        record_id = record["record_id"]

        # Step 2: Fetch all MARC fields for that record
        fields_query = (
            f"{SUPABASE_URL}/rest/v1/marc_fields"
            f"?record_id=eq.{record_id}&select=tag,field_value"
        )
        fields_res = requests.get(fields_query, headers=headers)
        fields = fields_res.json()

        # Step 3: Extract key details by tag
        tag_map = {f["tag"]: f["field_value"] for f in fields}
        book_data = {
            "control_no": control_no,
            "title": tag_map.get("245", ""),
            "author": tag_map.get("100", ""),
            "edition": tag_map.get("250", ""),
            "publisher": record.get("publishers", {}).get("name", ""),
            "isbn": tag_map.get("020", ""),
            "published_date": tag_map.get("260", ""),
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
        }

        return jsonify({"success": True, "data": book_data}), 200

    except Exception as e:
        print("Error fetching book:", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/get-book-by-title/<string:title>', methods=['GET'])
def get_book_by_title(title):
    """Fetch all books that match the given title keyword"""
    try:
        url = (
            f"{SUPABASE_URL}/rest/v1/marc_fields?"
            f"tag=eq.245&field_value=ilike.%25{title}%25&"
            f"select=record_id,field_value,marc_records(control_no,created_at,publishers(name))"
        )
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            data = res.json()
            if len(data) == 0:
                return jsonify({"success": False, "message": "No books found"}), 404

            books = []
            for record in data:
                record_id = record["record_id"]
                author_res = requests.get(
                    f"{SUPABASE_URL}/rest/v1/marc_fields?record_id=eq.{record_id}&tag=eq.100&select=field_value",
                    headers=headers
                )
                author = author_res.json()[0]['field_value'] if author_res.json() else ""

                edition_res = requests.get(
                    f"{SUPABASE_URL}/rest/v1/marc_fields?record_id=eq.{record_id}&tag=eq.250&select=field_value",
                    headers=headers
                )
                edition = edition_res.json()[0]['field_value'] if edition_res.json() else ""

                books.append({
                    "title": record["field_value"],
                    "control_no": record["marc_records"]["control_no"],
                    "publisher": record["marc_records"]["publishers"],
                    "created_at": record["marc_records"]["created_at"],
                    "author": author,
                    "edition": edition
                })

            return jsonify({"success": True, "data": books})
        return jsonify({"success": False, "message": "Book not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/update-book/<string:control_no>', methods=['POST'])
def update_book_info(control_no):
    """Update changed fields for a specific book (with IST timestamp)"""
    try:
        update_data = request.json or {}
        update_time = get_indian_time().isoformat()

        # Ensure the updated_at field is set with IST
        update_data['updated_at'] = update_time

        # 🧠 Step 1: Get the record_id for this control number
        record_res = requests.get(
            f"{SUPABASE_URL}/rest/v1/marc_records?control_no=eq.{control_no}&select=record_id,publisher_id",
            headers=headers
        )
        if record_res.status_code != 200 or not record_res.json():
            return jsonify({"success": False, "message": "Record not found"}), 404

        record_data = record_res.json()[0]
        record_id = record_data["record_id"]
        current_publisher_id = record_data.get("publisher_id")

        # 🧠 Step 2: For each field, update only if it changed
        # Define tag mappings
        tag_map = {
            "title": "245",
            "author": "100",
            "edition": "250",
            "isbn": "020",
            "published_date": "260",
            "publisher": "264"  # Added publisher to tag map
        }

        for key, tag in tag_map.items():
            if key in update_data and update_data[key]:
                payload = {"field_value": update_data[key]}
                update_response = requests.patch(
                    f"{SUPABASE_URL}/rest/v1/marc_fields?record_id=eq.{record_id}&tag=eq.{tag}",
                    headers=headers,
                    json=payload
                )
                if update_response.status_code in [200, 204]:
                    print(f"✅ Updated MARC field {tag}: {update_data[key]}")
                else:
                    print(f"⚠️ Failed to update MARC field {tag}: {update_response.text}")

        # 🧠 Step 3: Update publisher in publishers table AND marc_fields (if changed)
        if "publisher" in update_data and update_data["publisher"]:
            pub_name = update_data["publisher"]
            new_publisher_id = current_publisher_id
            
            # Check if we're updating to the same publisher name
            if current_publisher_id:
                # Get current publisher name
                current_pub_res = requests.get(
                    f"{SUPABASE_URL}/rest/v1/publishers?publisher_id=eq.{current_publisher_id}",
                    headers=headers
                )
                if current_pub_res.status_code == 200 and current_pub_res.json():
                    current_pub_name = current_pub_res.json()[0]["name"]
                    if current_pub_name == pub_name:
                        # Same publisher name, no need to update
                        print(f"✅ Publisher name unchanged: {pub_name}")
                    else:
                        # Different publisher name - update existing publisher
                        update_pub_res = requests.patch(
                            f"{SUPABASE_URL}/rest/v1/publishers?publisher_id=eq.{current_publisher_id}",
                            headers=headers,
                            json={"name": pub_name}
                        )
                        if update_pub_res.status_code in [200, 204]:
                            print(f"✅ Updated existing publisher: {current_pub_name} -> {pub_name}")
                        else:
                            print(f"⚠️ Failed to update publisher: {update_pub_res.text}")
                else:
                    # Couldn't get current publisher, create new one
                    new_pub_res = requests.post(
                        f"{SUPABASE_URL}/rest/v1/publishers",
                        headers=headers,
                        json={"name": pub_name}
                    )
                    if new_pub_res.status_code in [200, 201]:
                        new_publisher_id = new_pub_res.json()[0]["publisher_id"]
                        print(f"✅ Created new publisher: {pub_name}")
            else:
                # No current publisher ID, create new one
                new_pub_res = requests.post(
                    f"{SUPABASE_URL}/rest/v1/publishers",
                    headers=headers,
                    json={"name": pub_name}
                )
                if new_pub_res.status_code in [200, 201]:
                    new_publisher_id = new_pub_res.json()[0]["publisher_id"]
                    print(f"✅ Created new publisher: {pub_name}")

            # Update marc_records with publisher_id (if we have a valid publisher_id)
            if new_publisher_id:
                update_record_res = requests.patch(
                    f"{SUPABASE_URL}/rest/v1/marc_records?control_no=eq.{control_no}",
                    headers=headers,
                    json={"publisher_id": new_publisher_id, "updated_at": update_time}
                )
                if update_record_res.status_code in [200, 204]:
                    print(f"✅ Updated marc_records with publisher_id: {new_publisher_id}")
            
            # Also ensure the MARC field 264 is updated with the publisher name
            marc_publisher_res = requests.patch(
                f"{SUPABASE_URL}/rest/v1/marc_fields?record_id=eq.{record_id}&tag=eq.264",
                headers=headers,
                json={"field_value": pub_name}
            )
            if marc_publisher_res.status_code in [200, 204]:
                print(f"✅ Updated MARC field 264 with publisher: {pub_name}")
        else:
            # Update only timestamp if no publisher change
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/marc_records?control_no=eq.{control_no}",
                headers=headers,
                json={"updated_at": update_time}
            )

        return jsonify({"success": True, "message": "✅ Book updated successfully (IST time recorded)."}), 200

    except Exception as e:
        print("Error updating book:", e)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/books-list', methods=['GET'])
def get_books_list():
    """Fetch minimal list of books (title + control_no + created_at) for deletion page"""
    try:
        # Fetch record_id, control_no, created_at, and title (tag 245)
        url = (
            f"{SUPABASE_URL}/rest/v1/marc_fields?"
            "tag=eq.245&select=field_value,record_id,"
            "marc_records(control_no,created_at)"
            "&order=marc_records(created_at).desc"
        )
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if len(data) > 0:
                # Rename keys for easier use
                books = []
                for record in data:
                    title = record.get("field_value", "Unknown Title")
                    rec = record.get("marc_records", {})
                    books.append({
                        "title": title,
                        "control_no": rec.get("control_no"),
                        "created_at": rec.get("created_at")
                    })
                return jsonify({"success": True, "data": books})
        return jsonify({"success": False, "message": "No books found"}), 404

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/delete-books', methods=['POST'])
def delete_books():
    """Delete one or multiple books"""
    try:
        data = request.json
        control_nos = data.get("control_nos", [])
        if not control_nos:
            return jsonify({"success": False, "message": "No books selected"}), 400

        deleted = []
        for control_no in control_nos:
            response = requests.delete(
                f"{SUPABASE_URL}/rest/v1/marc_records?control_no=eq.{control_no}",
                headers=headers
            )
            if response.status_code in [200, 204]:
                deleted.append(control_no)

        return jsonify({
            "success": True,
            "deleted": deleted,
            "message": f"{len(deleted)} book(s) deleted successfully"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/dashboard')
def dashboard():
    # Serve the same index.html file (the SPA)
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
    
