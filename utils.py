# utils.py
import uuid
import datetime
import json
import os
from typing import List, Dict, Any, Optional
from io import BytesIO
import PyPDF2
import docx
import pytesseract
from PIL import Image
from config import Config

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(file_bytes: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_file = BytesIO(file_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    @staticmethod
    def extract_text_from_docx(file_bytes: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            doc_file = BytesIO(file_bytes)
            doc = docx.Document(doc_file)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            print(f"Error extracting text from DOCX: {e}")
            return ""

    @staticmethod
    def extract_text_from_image(file_bytes: bytes) -> str:
        """Extract text from image using OCR"""
        try:
            img = Image.open(BytesIO(file_bytes))
            return pytesseract.image_to_string(img)
        except Exception as e:
            print(f"Error extracting text from image: {e}")
            return ""

    @staticmethod
    def process_file(file_data: bytes, filename: str, file_type: str) -> str:
        """Process file based on type and extract text"""
        file_extension = filename.split('.')[-1].lower()
        
        if file_type == 'document':
            if file_extension == 'pdf':
                return DocumentProcessor.extract_text_from_pdf(file_data)
            elif file_extension in ['doc', 'docx']:
                return DocumentProcessor.extract_text_from_docx(file_data)
            elif file_extension == 'txt':
                return file_data.decode('utf-8')
        elif file_type == 'image':
            return DocumentProcessor.extract_text_from_image(file_data)
            
        return ""

    @staticmethod
    def create_chunks(text: str, max_chunk_size: int = Config.MAX_CHUNK_SIZE) -> List[str]:
        """Split text into chunks of appropriate size"""
        chunks = []
        
        if len(text) <= max_chunk_size:
            return [text]
            
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph would exceed max size
            if len(current_chunk) + len(para) < max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                # If current chunk is already substantial, add it to chunks
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If paragraph itself is too large, split it by sentences
                if len(para) > max_chunk_size:
                    sentences = para.split('. ')
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) < max_chunk_size:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + ". "
                else:
                    current_chunk = para + "\n\n"
        
        # Add any remaining text
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

class DocumentManager:
    @staticmethod
    def load_documents() -> List[Dict[str, Any]]:
        """Load document metadata from file"""
        if os.path.exists(Config.DOCUMENTS_FILE):
            with open(Config.DOCUMENTS_FILE, 'r') as f:
                return json.load(f)
        return []
        
    @staticmethod
    def save_documents(documents: List[Dict[str, Any]]) -> None:
        """Save document metadata to file"""
        with open(Config.DOCUMENTS_FILE, 'w') as f:
            json.dump(documents, f)
            
    @staticmethod
    def create_document_metadata(filename: str, file_type: str, file_path: str, size_bytes: int) -> Dict[str, Any]:
        """Create document metadata"""
        return {
            "id": str(uuid.uuid4()),
            "filename": filename,
            "file_type": file_type,
            "file_path": file_path,
            "upload_date": datetime.datetime.now().isoformat(),
            "size_bytes": size_bytes
        }

class PromptBuilder:
    @staticmethod
    def build_rag_prompt(query: str, contexts: List[str]) -> str:
        """Build prompt for RAG using Gemini"""
        context_text = "\n\n".join(contexts) if contexts else ""
        
        if context_text:
            return f"""Answer the following question based on the provided context, 
            don't answer questions not related
            with law
              and cite your sources related to your answer, 
            first state the related law and proceed to give your further opinion.
              If the context doesn't contain 
            relevant information, just give your own opinion and wrap that specific opinion in 
            <disclaimer>opinion...</disclaimer>
              and say you don't have enough information to answer accurately.
            
            Context:
            {context_text}
            
            Question: {query}
            
            Answer:"""
        else:
            return f"""Answer the following question: {query}
            If you don't have information about this topic, please state that clearly."""