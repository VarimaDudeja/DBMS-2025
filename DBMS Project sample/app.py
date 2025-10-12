from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import io
import pandas as pd
from werkzeug.utils import secure_filename
from ExtractingInfo import process_two_images_and_save
import tempfile
import requests  # Added missing import

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
        
        print(f"Processing files: {image1.filename}, {image2.filename}")
        
        if allowed_file(image1.filename) and allowed_file(image2.filename):
            # Save uploaded files temporarily
            filename1 = secure_filename(image1.filename)
            filename2 = secure_filename(image2.filename)
            
            filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
            
            image1.save(filepath1)
            image2.save(filepath2)
            
            print("Files saved, starting processing...")
            
            # Process images and save to Supabase database
            success, message, book_data = process_two_images_and_save(filepath1, filepath2)
            
            print(f"Processing result: {success}, {message}")
            
            # Clean up uploaded files
            try:
                os.remove(filepath1)
                os.remove(filepath2)
            except OSError as e:
                print(f"Warning: Could not remove temporary files: {e}")
            
            return jsonify({
                'success': success,
                'message': message,
                'book_data': book_data
            })
        else:
            return jsonify({'success': False, 'message': 'Invalid file type'}), 400
            
    except Exception as e:
        print(f"Error in process-book: {str(e)}")
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

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
    try:
        # Get all books via REST API
        books_response = requests.get(
            f"{SUPABASE_URL}/rest/v1/marc_fields?tag=eq.245&select=field_value,record_id,marc_records(control_no,created_at,publishers(name))",
            headers=headers
        )
        
        if books_response.status_code != 200:
            return jsonify({'success': False, 'message': 'Failed to fetch data'})
        
        books_data = books_response.json()
        excel_data = []
        
        for book in books_data:
            record_id = book['record_id']
            title = book['field_value']
            record = book.get('marc_records', {})
            
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
            
            excel_data.append({
                'Title': title,
                'Authors': ', '.join(authors) if authors else 'Unknown Author',
                'Publisher': record.get('publishers', {}).get('name', 'Unknown Publisher') if isinstance(record.get('publishers'), dict) else 'Unknown Publisher',
                'Subjects': ', '.join(subjects) if subjects else 'General',
                'Control_No': record.get('control_no', ''),
                'Created_At': record.get('created_at', '')
            })
        
        # Create DataFrame and Excel file
        df = pd.DataFrame(excel_data)
        excel_file = io.BytesIO()
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Books')
        excel_file.seek(0)
        
        return send_file(
            excel_file,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'libraScan_books_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )
        
    except Exception as e:
        print(f"Excel download error: {str(e)}")
        return jsonify({'success': False, 'message': f'Excel download error: {str(e)}'})

@app.route('/api/dashboard-stats', methods=['GET'])
def get_dashboard_stats():
    try:
        # Get total books count
        books_response = requests.get(
            f"{SUPABASE_URL}/rest/v1/marc_records?select=count",
            headers=headers
        )
        
        total_books = 0
        if books_response.status_code == 200:
            count_data = books_response.json()
            if isinstance(count_data, list) and len(count_data) > 0:
                total_books = count_data[0]['count']
            elif isinstance(count_data, dict):
                total_books = count_data.get('count', 0)
        
        # Get recent books
        recent_response = requests.get(
            f"{SUPABASE_URL}/rest/v1/marc_records?select=record_id,created_at&order=created_at.desc&limit=5",
            headers=headers
        )
        
        recent_books = []
        if recent_response.status_code == 200:
            recent_data = recent_response.json()
            for record in recent_data:
                # Get title
                title_response = requests.get(
                    f"{SUPABASE_URL}/rest/v1/marc_fields?record_id=eq.{record['record_id']}&tag=eq.245",
                    headers=headers
                )
                title = "Unknown Title"
                if title_response.status_code == 200:
                    title_data = title_response.json()
                    if title_data:
                        title = title_data[0]['field_value']
                
                # Get authors
                authors_response = requests.get(
                    f"{SUPABASE_URL}/rest/v1/record_authors?record_id=eq.{record['record_id']}&select=authors(name)",
                    headers=headers
                )
                authors = []
                if authors_response.status_code == 200:
                    authors_data = authors_response.json()
                    authors = [author['authors']['name'] for author in authors_data if 'authors' in author and author['authors']]
                
                recent_books.append({
                    'title': title,
                    'authors': ', '.join(authors) if authors else 'Unknown Author',
                    'time': record.get('created_at', '')[:16] if record.get('created_at') else ''
                })
        
        return jsonify({
            'success': True,
            'stats': {
                'total_books': total_books,
                'today_records': total_books,  # Simplified for now
                'recent_books': recent_books
            }
        })
        
    except Exception as e:
        print(f"Stats error: {str(e)}")
        return jsonify({'success': False, 'message': f'Stats error: {str(e)}'})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'LibraScan API is running'})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')