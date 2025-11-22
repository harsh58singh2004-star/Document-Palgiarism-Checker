from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import uuid
import logging

# Optional docx support
try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'docx'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = (text or "").lower()
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def read_file(filepath):
    """Read content from file. Supports .txt and .docx (if python-docx installed)."""
    ext = filepath.rsplit('.', 1)[-1].lower()
    if ext == 'txt':
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read()
    elif ext == 'docx':
        if not DOCX_SUPPORT:
            raise RuntimeError("python-docx is required to read .docx files. Install with: pip install python-docx")
        doc = Document(filepath)
        paragraphs = [p.text for p in doc.paragraphs]
        return '\n'.join(paragraphs)
    else:
        # Shouldn't happen due to prior validation, but keep safe
        raise ValueError("Unsupported file type: " + ext)


def calculate_similarity(text1, text2):
    """Calculate cosine similarity using TF-IDF vectorization"""
    # Preprocess texts
    text1_clean = preprocess_text(text1)
    text2_clean = preprocess_text(text2)
    
    if not text1_clean or not text2_clean:
        return 0.0

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Generate TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform([text1_clean, text2_clean])
    
    # Calculate cosine similarity
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return score * 100  # Convert to percentage


def find_plagiarized_sentences(text1, text2, threshold=0.7):
    """Find similar sentences between two documents"""
    # Split texts into sentences
    sentences1 = [s.strip() for s in re.split(r'[.!?]+', text1) if s.strip()]
    sentences2 = [s.strip() for s in re.split(r'[.!?]+', text2) if s.strip()]
    
    plagiarized_parts = []
    
    for i, sent1 in enumerate(sentences1):
        if len(sent1) < 20:  # Skip very short sentences
            continue
            
        for j, sent2 in enumerate(sentences2):
            if len(sent2) < 20:
                continue
            
            # Calculate similarity ratio for sentences
            ratio = difflib.SequenceMatcher(None, sent1.lower(), sent2.lower()).ratio()
            
            if ratio >= threshold:
                plagiarized_parts.append({
                    'doc1_sentence': sent1,
                    'doc2_sentence': sent2,
                    'similarity': round(ratio * 100, 2),
                    'doc1_index': i,
                    'doc2_index': j
                })
    
    return plagiarized_parts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_plagiarism():
    try:
        # Check if files are present
        if 'file1' not in request.files or 'file2' not in request.files:
            return jsonify({'error': 'Both files are required'}), 400
        
        file1 = request.files['file1']
        file2 = request.files['file2']
        
        # Validate files
        if not file1 or not file2 or file1.filename == '' or file2.filename == '':
            return jsonify({'error': 'Please select both files'}), 400
        
        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return jsonify({'error': 'Only .txt and .docx files are supported'}), 400
        
        # If .docx support is missing, reject .docx uploads with clear message
        ext1 = file1.filename.rsplit('.', 1)[-1].lower()
        ext2 = file2.filename.rsplit('.', 1)[-1].lower()
        if (ext1 == 'docx' or ext2 == 'docx') and not DOCX_SUPPORT:
            return jsonify({'error': 'python-docx is required to process .docx files. Install with: pip install python-docx'}), 400
        
        # Save files with unique prefixes to avoid collisions
        filename1 = f"{uuid.uuid4().hex}_{secure_filename(file1.filename)}"
        filename2 = f"{uuid.uuid4().hex}_{secure_filename(file2.filename)}"
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        
        file1.save(filepath1)
        file2.save(filepath2)
        
        # Read file contents
        text1 = read_file(filepath1)
        text2 = read_file(filepath2)
        
        # Calculate overall similarity
        similarity_score = calculate_similarity(text1, text2)
        
        # Find plagiarized sentences
        plagiarized_parts = find_plagiarized_sentences(text1, text2)
        
        # Clean up uploaded files
        try:
            if os.path.exists(filepath1):
                os.remove(filepath1)
            if os.path.exists(filepath2):
                os.remove(filepath2)
        except OSError as e:
            logger.warning("Failed to remove uploaded files: %s", e)
        
        return jsonify({
            'similarity_score': round(similarity_score, 2),
            'plagiarized_parts': plagiarized_parts[:10],  # Return top 10 matches
            'total_matches': len(plagiarized_parts),
            'doc1_content': text1,
            'doc2_content': text2
        })
        
    except Exception as e:
        logger.exception("Error in plagiarism check")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)