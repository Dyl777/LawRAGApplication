# rag_backend.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from typing import List, Dict, Any, Optional

# Import local modules
from config import Config
from utils import DocumentProcessor, DocumentManager, PromptBuilder

# RAG components
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup directories
Config.setup_directories()

# Initialize Google Gemini
genai.configure(api_key=Config.GOOGLE_API_KEY)

# Initialize embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=Config.GOOGLE_API_KEY
)

# Gemini pro model for text generation
generation_model = genai.GenerativeModel('gemini-2.0-flash')

# Load documents metadata
documents = DocumentManager.load_documents()

# Initialize or load FAISS vector store
try:
    vector_store = FAISS.load_local(Config.FAISS_INDEX_FOLDER, embedding_model)
    print("Loaded existing FAISS index")
except Exception as e:
    print(f"Creating new FAISS index: {e}")
    vector_store = FAISS.from_texts(["Initialize vector store"], embedding_model)
    vector_store.save_local(Config.FAISS_INDEX_FOLDER)

def add_to_vector_store(text: str, doc_id: str) -> bool:
    """Add document text to vector store"""
    try:
        # Split text into smaller chunks
        chunks = DocumentProcessor.create_chunks(text)
            
        # Add metadata to each chunk
        texts_with_metadata = []
        for i, chunk in enumerate(chunks):
            texts_with_metadata.append((chunk, {"doc_id": doc_id, "chunk_id": i}))
        
        # Add to vector store
        vector_store.add_texts([t[0] for t in texts_with_metadata], 
                               metadatas=[t[1] for t in texts_with_metadata])
        vector_store.save_local(Config.FAISS_INDEX_FOLDER)
        return True
    except Exception as e:
        print(f"Error adding to vector store: {e}")
        return False

#@app.route('/upload', methods=['POST'])
#def upload_file():
    #"""Upload and process a file"""
    #try:
        #data = request.json
        #if not data or 'file_data' not in data:
            #return jsonify({"error": "No file data provided"}), 400
            
        #filename = data.get('filename', f"file_{os.urandom(4).hex()}")
        #file_type = data.get('file_type', 'document')
        #file_data = base64.b64decode(data['file_data'])
        
        # Process file to extract text
        #extracted_text = DocumentProcessor.process_file(file_data, filename, file_type)
        #if not extracted_text:
            #return jsonify({"error": f"Could not extract text from {filename}"}), 400
            
        # Save file to uploads directory
        #file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        #with open(file_path, 'wb') as f:
            #f.write(file_data)
            
        # Create document metadata
        #doc_info = DocumentManager.create_document_metadata(
            #filename, file_type, file_path, len(file_data)
        #)
        
        # Add document metadata
        #documents.append(doc_info)
        #DocumentManager.save_documents(documents)
        
        # Add to vector store
        #if add_to_vector_store(extracted_text, doc_info["id"]):
            #return jsonify({
                #"message": "File uploaded and processed successfully", 
                #"doc_id": doc_info["id"]
            #}), 200
        #else:
            #return jsonify({"error": "Error adding document to vector store"}), 500
            
    #except Exception as e:
        #return jsonify({"error": f"Error processing upload: {str(e)}"}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process a file"""
    try:
        data = request.json
        print(data)
        if not data or 'file_data' not in data:
            return jsonify({"error": "No file data provided"}), 400
            
        filename = data.get('filename', f"file_{os.urandom(4).hex()}")
        file_type = data.get('file_type', 'document')

        try:
            file_data = base64.b64decode(data['file_data'])
        except Exception as e:
            return jsonify({"error": f"Base64 decoding failed: {str(e)}"}), 400
        
        # Process file to extract text
        extracted_text = DocumentProcessor.process_file(file_data, filename, file_type)
        if not extracted_text:
            return jsonify({"error": f"Could not extract text from {filename}"}), 400
            
        # Save file to uploads directory
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        with open(file_path, 'wb') as f:
            f.write(file_data)
            
        # Create document metadata
        doc_info = DocumentManager.create_document_metadata(
            filename, file_type, file_path, len(file_data)
        )
        
        # Add document metadata
        documents.append(doc_info)
        DocumentManager.save_documents(documents)
        
        # Add to vector store
        if add_to_vector_store(extracted_text, doc_info["id"]):
            return jsonify({
                "message": "File uploaded and processed successfully", 
                "doc_id": doc_info["id"]
            }), 200
        else:
            return jsonify({"error": "Error adding document to vector store"}), 500
            
    except Exception as e:
        return jsonify({"error": f"Error processing upload: {str(e)}"}), 500



@app.route('/documents', methods=['GET'])
def get_documents():
    """Get all documents in the knowledge base"""
    return jsonify(documents), 200

@app.route('/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document from the knowledge base"""
    global documents
    # Update global vector store
    global vector_store
    
    # Find document by ID
    doc_to_delete = None
    for doc in documents:
        if doc['id'] == doc_id:
            doc_to_delete = doc
            break
            
    if not doc_to_delete:
        return jsonify({"error": "Document not found"}), 404
        
    # Remove physical file
    try:
        if os.path.exists(doc_to_delete['file_path']):
            os.remove(doc_to_delete['file_path'])
    except Exception as e:
        print(f"Error removing file: {e}")
        
    # Remove from documents list
    documents = [doc for doc in documents if doc['id'] != doc_id]
    DocumentManager.save_documents(documents)
    
    # For FAISS, we'll rebuild the index without this document's chunks
    try:
        # Get all texts and metadatas except for the deleted document
        results = vector_store.similarity_search_with_score("", k=10000)
        filtered_texts = []
        filtered_metadatas = []
        
        for doc, _ in results:
            if doc.metadata.get("doc_id") != doc_id:
                filtered_texts.append(doc.page_content)
                filtered_metadatas.append(doc.metadata)
                
        # Recreate vector store if we have documents
        if filtered_texts:
            new_vector_store = FAISS.from_texts(filtered_texts, embedding_model, metadatas=filtered_metadatas)
            new_vector_store.save_local(Config.FAISS_INDEX_FOLDER)
            
            
            vector_store = new_vector_store
        else:
            # No documents left, create empty vector store
            new_vector_store = FAISS.from_texts(["Initialize vector store"], embedding_model)
            new_vector_store.save_local(Config.FAISS_INDEX_FOLDER)
            vector_store = new_vector_store
        
        return jsonify({"message": "Document deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Error updating vector store: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Process a chat query using RAG"""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
            
        query = data['query']
        
        # Search for relevant documents
        search_results = vector_store.similarity_search_with_score(
            query, 
            k=Config.DEFAULT_SEARCH_RESULTS
        )
        
        # Extract relevant contexts and their sources
        contexts = []
        sources = []
        
        for doc, score in search_results:
            if score < Config.RELEVANCE_THRESHOLD:  # Filter by relevance score
                contexts.append(doc.page_content)
                
                # Find source document
                doc_id = doc.metadata.get("doc_id")
                for document in documents:
                    if document['id'] == doc_id and document['filename'] not in sources:
                        sources.append(document['filename'])
                        break
        
        # Build prompt with contexts
        prompt = PromptBuilder.build_rag_prompt(query, contexts)
            
        # Generate response using Gemini
        response = generation_model.generate_content(prompt)
        
        return jsonify({
            "response": response.text,
            "sources": sources
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Error processing query: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)