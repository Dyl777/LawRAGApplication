from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# Create an upload folder if it doesn't exist
upload_folder = 'uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Set the upload folder
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'jpg', 'jpeg', 'png', 'doc', 'docx', 'txt'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route for uploading files
@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('files')
    
    if not files:
        return jsonify({'error': 'No selected files'}), 400
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            return jsonify({'error': f'File type not allowed: {file.filename}'}), 400
    
    return jsonify({'message': 'Files uploaded successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


