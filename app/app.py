"""
app.py

Main Flask application, routes, and SocketIO setup.
"""

import config
import os
import uuid
import logging

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

import openai
import nltk
from sentence_transformers import SentenceTransformer
import qdrant_client

# Import our modules
from pdf_processor import PDFProcessor
from qdrant_manager import QdrantManager

# -----------------------------------------------------------------------------
#                         NLTK and Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('punkt_tab')
nltk.download("punkt")

# -----------------------------------------------------------------------------
#                           Flask & SocketIO Setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25
)

# -----------------------------------------------------------------------------
#                       Model and Qdrant Configuration
# -----------------------------------------------------------------------------

model = SentenceTransformer(config.SENTENCE_TRANSFORMER)
qdrant = qdrant_client.QdrantClient(config.DB_ADDR+":"+str(config.DB_PORT))

# Create our Qdrant manager
qdrant_manager = QdrantManager(qdrant, model)

# -----------------------------------------------------------------------------
#                      OpenAI Configuration (unchanged)
# -----------------------------------------------------------------------------
OPEN_AI_ORG_ID = os.getenv('OPEN_AI_ORG_ID')
OPEN_AI_API_KEY = os.getenv('OPEN_AI_API_KEY')
openai.organization = OPEN_AI_ORG_ID

# Keep the custom client usage as requested
open_ai_client = openai.Client(api_key=OPEN_AI_API_KEY)

# -----------------------------------------------------------------------------
#                           Upload Folder Config
# -----------------------------------------------------------------------------
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit

# -----------------------------------------------------------------------------
#                                Routes
# -----------------------------------------------------------------------------
@app.route('/')
def home():
    """
    Renders the home page with a list of existing Qdrant collections (projects).
    """
    try:
        projects = qdrant_manager.get_collections()
        return render_template('index.html', projects=projects)
    except Exception as e:
        logger.error(f"Error fetching projects: {e}")
        return jsonify({"error": f"Error fetching projects: {str(e)}"}), 500


@app.route('/delete_project', methods=['POST'])
def delete_project():
    """
    Process the Delete Project form submission and remove the specified collection from Qdrant.
    """
    try:
        project_id = request.form.get('project_id')
        if not project_id:
            return jsonify({"error": "Project ID is required."}), 400

        existing = [c.name for c in qdrant_manager.get_collections()]
        if project_id not in existing:
            return jsonify({"error": f"Project '{project_id}' does not exist."}), 404

        qdrant_manager.delete_collection(project_id)
        return jsonify({"message": f"Project '{project_id}' has been successfully deleted."}), 200
    except Exception as e:
        logger.error(f"Error deleting project: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/add_project', methods=['POST'])
def add_project():
    """
    Create a new Qdrant collection (project).
    """
    project_id = request.form.get('project_id')
    if not project_id:
        return jsonify({"error": "Project ID is required."}), 400

    try:
        qdrant_manager.create_collection(project_id)
        return jsonify({"message": f"Project '{project_id}' has been successfully created."}), 200
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/upload_pdf', methods=['GET'])
def show_upload_pdf_form():
    """
    Renders the PDF upload form.
    """
    project_id = request.args.get('project_id', default='', type=str)
    return render_template('upload_pdf.html', project_id=project_id)


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """
    Handle PDF upload and start background processing via SocketIO.
    """
    project_id = request.form.get('project_id')
    pdf_file = request.files.get('file')
    socket_id = request.form.get('socket_id')

    logger.info(f"Received upload request for project_id: {project_id} from socket_id: {socket_id}")

    if not project_id:
        logger.warning("No Project ID provided.")
        return jsonify({"error": "Project ID is required."}), 400
    if not pdf_file:
        logger.warning("No PDF file uploaded.")
        return jsonify({"error": "No PDF uploaded."}), 400
    if not socket_id:
        logger.warning("No Socket ID provided.")
        return jsonify({"error": "No socket ID provided for progress updates."}), 400
    if not pdf_file.filename.lower().endswith('.pdf'):
        logger.warning("Uploaded file is not a PDF.")
        return jsonify({"error": "Uploaded file is not a PDF."}), 400

    try:
        filename = secure_filename(pdf_file.filename)
        unique_id = str(uuid.uuid4())
        saved_filename = f"{unique_id}_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        pdf_file.save(save_path)
        logger.info(f"Saved PDF file to {save_path}")

        # Start background task for processing
        socketio.start_background_task(
            process_pdf,
            project_id,
            pdf_file.filename,
            save_path,
            socket_id
        )
        logger.info(f"Started background task for processing PDF: {save_path}")

        return jsonify({"message": "PDF upload successful and processing started."}), 200
    except Exception as e:
        error_message = f"Error processing PDF: {str(e)}"
        logger.error(error_message)
        return jsonify({"error": error_message}), 500


def process_pdf(project_id: str, original_file_name: str, pdf_path: str, socket_id: str):
    """
    Background thread function to chunk, embed, and upload the PDF content to Qdrant.
    """
    try:
        socketio.emit('processing_progress', {'progress': 0}, room=socket_id)
        socketio.emit('status', {'message': 'Starting PDF processing...'}, room=socket_id)

        # Ensure Qdrant collection
        qdrant_manager.ensure_collection_exists(project_id)

        # Chunk PDF
        pdf_chunks = PDFProcessor.chunk_pdf_text(
            pdf_file_path=pdf_path,
            original_file_name=original_file_name,
            socketio_instance=socketio,
            socket_id=socket_id,
            max_words=300,
            overlap_sentences=1
        )

        if not pdf_chunks:
            socketio.emit('processing_error', {'message': 'No text could be extracted from PDF'}, room=socket_id)
            return

        socketio.emit('status', {'message': 'Generating embeddings...'}, room=socket_id)

        # Embed chunk texts
        chunk_texts = [chunk['text'] for chunk in pdf_chunks]
        embeddings = model.encode(chunk_texts).astype('float32').tolist()

        socketio.emit('status', {'message': 'Uploading to Qdrant...'}, room=socket_id)

        # Prepare points
        points = []
        for chunk, embedding in zip(pdf_chunks, embeddings):
            points.append({
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {
                    "pdf_name": chunk['pdf_name'],
                    "page": chunk['page'],
                    "chunk_id": chunk['chunk_id'],
                    "text": chunk['text']
                }
            })

        # Upsert
        qdrant_manager.upsert_points(project_id, points)

        socketio.emit('processing_progress', {'progress': 100}, room=socket_id)
        socketio.emit('processing_complete', {
            'message': 'PDF processed and uploaded successfully'
        }, room=socket_id)

    except Exception as e:
        error_message = f"Error processing PDF: {str(e)}"
        logger.error(error_message)
        socketio.emit('processing_error', {'message': error_message}, room=socket_id)

    finally:
        # Delete the uploaded file
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                logger.info(f"Deleted file: {pdf_path}")
            else:
                logger.warning(f"File not found for deletion: {pdf_path}")
        except Exception as file_del_error:
            logger.error(f"Error deleting file '{pdf_path}': {file_del_error}")


@app.route('/query_project', methods=['GET'])
def show_query_project_form():
    """
    Renders a form for querying a project.
    """
    project_id = request.args.get('project_id', default='', type=str)
    return render_template('query_project.html', project_id=project_id)


@app.route('/query_project', methods=['POST'])
def query_project():
    """
    Handle queries against a Qdrant project.
    """
    project_id = request.form['project_id']
    question = request.form['question']
    threshold = float(request.form['threshold'])

    if not project_id or not question:
        return jsonify({"error": "Project ID and question are required."}), 400

    try:
        # Embed the user question
        question_embedding = model.encode([question], convert_to_numpy=True).astype('float32')[0]
        results = qdrant.search(
            collection_name=project_id,
            query_vector=question_embedding.tolist(),
            limit=20
        )

        filtered_results = [
            {"score": r.score, "chunk": r.payload}
            for r in results if r.score >= threshold
        ]

        if not filtered_results:
            return jsonify({"answer": "No relevant information found above the threshold"})

        # Build and answer
        prompt = build_prompt(question, [res['chunk'] for res in filtered_results])
        answer = construct_an_answer(prompt)

        return render_template('query_results.html', answer=answer, results=filtered_results)

    except Exception as e:
        logger.error(f"Error querying project: {e}")
        return jsonify({"error": f"Error querying project: {str(e)}"}), 500


def build_prompt(user_question: str, retrieved_chunks: list) -> str:
    """
    Construct a prompt that includes the user question, retrieved chunks, 
    and instructions to answer only from these chunks.
    """
    prompt_context = "Excerpts:\n"
    for chunk in retrieved_chunks:
        pdf_name = chunk['pdf_name']
        page = chunk['page']
        chunk_id = chunk['chunk_id']
        reference_str = f"(PDF '{pdf_name}', page {page}, chunk {chunk_id})"
        prompt_context += f"--- {reference_str} ---\n{chunk['text']}\n\n"

    prompt_question = f"User question: {user_question}\n\n"
    return prompt_context + prompt_question


def construct_an_answer(prompt: str) -> str:
    """
    Uses the custom open_ai_client to get an answer from the LLM.
    """
    try:
        response = open_ai_client.chat.completions.create(
            model=config.OPEN_AI_MODEL,  # Example model; you can adjust as needed
            messages=[
                {
                    "role": "user",
                    "content": """
                        1. You are an AI assistant that strictly answers the user question using only the provided excerpts. 
                        2. You will create a holistic and deeply detailed answer using all the relevant excerpts. 
                        3. If the answer does not exist in the excerpts, say: 'I don't know from the provided text.' 
                        4. In your detailed formulated answer, cite the excerpts used in parentheses with ('<pdf_name>', page X, chunk Y).
                    """ + prompt
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error connecting to LLM: {e}")
        return 'Error connecting to LLM'


# -----------------------------------------------------------------------------
#                      Socket.IO Event Handlers
# -----------------------------------------------------------------------------
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")


# -----------------------------------------------------------------------------
#                                   Main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    socketio.run(app, debug=True, host=config.WEB_SERVER_HOST, port=config.WEB_SERVER_PORT)
