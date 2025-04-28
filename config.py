# config.py
import os

class Config:
    # API Keys
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', 'AIzaSyCopFldRoxw7xAL5fe3Rc2-RuvMMgIqntk')
    
    # File paths and directories
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    DB_FOLDER = os.path.join(BASE_DIR, 'vector_db')
    FAISS_INDEX_FOLDER = os.path.join(DB_FOLDER, 'faiss_index')
    DOCUMENTS_FILE = os.path.join(DB_FOLDER, 'documents.json')
    
    # RAG Configuration
    MAX_CHUNK_SIZE = 2000       # Maximum size of text chunks
    DEFAULT_SEARCH_RESULTS = 3  # Number of search results to return
    RELEVANCE_THRESHOLD = 1.0   # Threshold for relevant search results
    
    # HTTP Server Configuration
    DEBUG = os.environ.get('DEBUG', 'True') == 'True'
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    # Create necessary directories
    @classmethod
    def setup_directories(cls):
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.DB_FOLDER, exist_ok=True)
        os.makedirs(cls.FAISS_INDEX_FOLDER, exist_ok=True)

