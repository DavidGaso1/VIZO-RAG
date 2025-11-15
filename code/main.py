from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os
import logging
import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv  # type: ignore
from utils import load_yaml_config
from prompt_builder import build_prompt_from_config
from langchain_google_genai import ChatGoogleGenerativeAI # pyright: ignore[reportMissingImports]
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage # pyright: ignore[reportMissingImports]
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR, VECTOR_DB_DIR
from vector_db_ingest import get_db_collection, embed_documents, chunk_publication

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

logger = logging.getLogger()

# Configuration for auto-ingestion
WATCH_FOLDER = "data/documents"  # Folder to watch for new documents
INGESTION_CACHE_FILE = "data/.ingestion_cache.json"  # Track ingested files
SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.doc', '.md', '.json']

# Collection name - MUST match your ingestion script
COLLECTION_NAME = "Vizo_Product_Manual"


def setup_logging():
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(os.path.join(OUTPUTS_DIR, "rag_assistant_with_memory.log"))
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# Initialize at module level
load_dotenv()
setup_logging()

# To avoid tokenizer parallelism warning from huggingface
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load configurations
app_config = load_yaml_config(APP_CONFIG_FPATH)
prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)

# Use enhanced RAG prompt
rag_assistant_prompt = prompt_config["rag_assistant_prompt_enhanced"]

vectordb_params = app_config["vectordb"]
llm_model = app_config["llm"]

# Initialize LLM once
llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.7)

# Initialize collection - Use the same collection name and directory as your ingestion script
logging.info(f"Connecting to ChromaDB at: {VECTOR_DB_DIR}")
logging.info(f"Collection name: {COLLECTION_NAME}")

collection = get_db_collection(
    persist_directory=VECTOR_DB_DIR,
    collection_name=COLLECTION_NAME
)

logging.info(f"Collection loaded. Total documents: {collection.count()}")

# Conversation memory (in production, use sessions or database)
conversation_histories = {}

# Create watch folder if it doesn't exist
os.makedirs(WATCH_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(INGESTION_CACHE_FILE), exist_ok=True)


def get_file_hash(filepath: str) -> str:
    """Calculate MD5 hash of a file to detect changes."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logging.error(f"Error hashing file {filepath}: {e}")
        return ""


def load_ingestion_cache() -> dict:
    """Load the cache of previously ingested files."""
    if os.path.exists(INGESTION_CACHE_FILE):
        try:
            with open(INGESTION_CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading ingestion cache: {e}")
            return {}
    return {}


def save_ingestion_cache(cache: dict):
    """Save the ingestion cache."""
    try:
        with open(INGESTION_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving ingestion cache: {e}")


def get_supported_files(folder: str) -> list:
    """Get list of supported document files in the folder."""
    files = []
    
    for root, dirs, filenames in os.walk(folder):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                files.append(os.path.join(root, filename))
    
    return files


def read_document_content(filepath: str) -> str:
    """
    Read content from different document types.
    
    Args:
        filepath: Path to the document
        
    Returns:
        str: Document content
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        if ext == '.txt' or ext == '.md':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif ext == '.pdf':
            try:
                import PyPDF2
                with open(filepath, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            except ImportError:
                logging.error("PyPDF2 not installed. Install with: pip install PyPDF2")
                return ""
                
        elif ext in ['.docx', '.doc']:
            try:
                import docx2txt
                return docx2txt.process(filepath)
            except ImportError:
                logging.error("docx2txt not installed. Install with: pip install docx2txt")
                return ""
                
        elif ext == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                import json
                data = json.load(f)
                # Convert JSON to text representation
                return json.dumps(data, indent=2)
        else:
            logging.warning(f"Unsupported file type: {ext}")
            return ""
            
    except Exception as e:
        logging.error(f"Error reading {filepath}: {e}")
        return ""


def ingest_document(filepath: str) -> bool:
    """
    Ingest a single document into the vector database.
    Uses your existing chunk_publication and embed_documents functions.
    
    Args:
        filepath: Path to the document file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logging.info(f"Ingesting document: {filepath}")
        
        # Read document content
        content = read_document_content(filepath)
        if not content or len(content.strip()) == 0:
            logging.warning(f"No content extracted from {filepath}")
            return False
        
        logging.info(f"Read {len(content)} characters from {filepath}")
        
        # Chunk using your existing function
        chunks = chunk_publication(content, chunk_size=1000, chunk_overlap=200)
        if not chunks:
            logging.warning(f"No chunks created from {filepath}")
            return False
        
        logging.info(f"Created {len(chunks)} chunks from {filepath}")
        
        # Embed using your existing function
        embeddings = embed_documents(chunks)
        logging.info(f"Generated {len(embeddings)} embeddings")
        
        # Get the next available ID from collection
        next_id = collection.count()
        
        # Create IDs for the chunks
        filename = os.path.basename(filepath)
        safe_filename = filename.replace('.', '_').replace(' ', '_')
        ids = [f"{safe_filename}_chunk_{i}" for i in range(len(chunks))]
        
        # Create metadata for each chunk
        metadatas = [
            {
                'source': filepath,
                'filename': filename,
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            for i in range(len(chunks))
        ]
        
        # Add to collection using the same method as your insert_publications
        collection.add(
            embeddings=embeddings,
            ids=ids,
            documents=chunks,
            metadatas=metadatas
        )
        
        logging.info(f"✓ Successfully ingested {len(chunks)} chunks from {filepath}")
        return True
        
    except Exception as e:
        logging.error(f"✗ Error ingesting {filepath}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False


def check_and_ingest_new_files() -> dict:
    """
    Check for new or modified files and ingest them.
    Returns a dict with ingestion statistics.
    """
    logging.info("Checking for new files to ingest...")
    
    cache = load_ingestion_cache()
    files = get_supported_files(WATCH_FOLDER)
    
    new_files = []
    modified_files = []
    failed_files = []
    
    for filepath in files:
        file_hash = get_file_hash(filepath)
        if not file_hash:
            continue
        
        relative_path = os.path.relpath(filepath, WATCH_FOLDER)
        cached_hash = cache.get(relative_path, {}).get('hash')
        
        # Check if file is new or modified
        if cached_hash is None:
            # New file
            logging.info(f"New file detected: {relative_path}")
            if ingest_document(filepath):
                cache[relative_path] = {
                    'hash': file_hash,
                    'ingested_at': datetime.now().isoformat(),
                    'filepath': filepath
                }
                new_files.append(relative_path)
            else:
                failed_files.append(relative_path)
                
        elif cached_hash != file_hash:
            # Modified file
            logging.info(f"Modified file detected: {relative_path}")
            if ingest_document(filepath):
                cache[relative_path] = {
                    'hash': file_hash,
                    'ingested_at': datetime.now().isoformat(),
                    'filepath': filepath
                }
                modified_files.append(relative_path)
            else:
                failed_files.append(relative_path)
    
    # Save updated cache
    if new_files or modified_files:
        save_ingestion_cache(cache)
    
    stats = {
        'new_files': len(new_files),
        'modified_files': len(modified_files),
        'failed_files': len(failed_files),
        'total_tracked': len(cache),
        'new_file_list': new_files,
        'modified_file_list': modified_files,
        'failed_file_list': failed_files
    }
    
    if new_files or modified_files:
        logging.info(f"Ingestion complete: {stats['new_files']} new, {stats['modified_files']} modified")
    else:
        logging.info("No new files to ingest")
    
    return stats


def retrieve_relevant_documents(
    query: str,
    n_results: int = 5,
    threshold: float = 0.3,
) -> list[str]:
    """
    Query the ChromaDB database with a string query.

    Args:
        query (str): The search query string
        n_results (int): Number of results to return (default: 5)
        threshold (float): Threshold for the cosine similarity score (default: 0.3)

    Returns:
        list: List of relevant document strings
    """
    logging.info(f"Retrieving relevant documents for query: {query}")
    relevant_results = {
        "ids": [],
        "documents": [],
        "distances": [],
    }
    # Embed the query using the same model used for documents
    logging.info("Embedding query...")
    query_embedding = embed_documents([query])[0]  # Get the first (and only) embedding

    logging.info("Querying collection...")
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )

    logging.info("Filtering results...")
    keep_item = [False] * len(results["ids"][0])
    for i, distance in enumerate(results["distances"][0]):
        if distance < threshold:
            keep_item[i] = True

    for i, keep in enumerate(keep_item):
        if keep:
            relevant_results["ids"].append(results["ids"][0][i])
            relevant_results["documents"].append(results["documents"][0][i])
            relevant_results["distances"].append(results["distances"][0][i])

    return relevant_results["documents"]


def format_conversation_history(conversation_history: list, max_pairs: int = 3) -> str:
    """
    Format conversation history for context.
    
    Args:
        conversation_history: List of message tuples (question, answer)
        max_pairs: Maximum number of recent Q&A pairs to include
    
    Returns:
        Formatted conversation history string
    """
    if not conversation_history:
        return ""
    
    # Get only the most recent pairs
    recent_history = conversation_history[-max_pairs:]
    
    formatted = "\n\n=== CONVERSATION HISTORY ===\n"
    for i, (question, answer) in enumerate(recent_history, 1):
        formatted += f"\nPrevious Question {i}: {question}\n"
        formatted += f"Previous Answer {i}: {answer}\n"
    formatted += "\n=== END CONVERSATION HISTORY ===\n\n"
    
    return formatted


def respond_to_query(
    prompt_config: dict,
    query: str,
    llm: ChatGoogleGenerativeAI,
    conversation_history: list = None,
    n_results: int = 5,
    threshold: float = 0.3,
    use_memory: bool = True,
) -> str:
    """
    Respond to a query using the ChromaDB database with optional conversation memory.
    
    Args:
        prompt_config: Prompt configuration dictionary
        query: User's question
        llm: Language model instance
        conversation_history: List of (question, answer) tuples
        n_results: Number of documents to retrieve
        threshold: Similarity threshold for retrieval
        use_memory: Whether to include conversation history
    
    Returns:
        Response string
    """
    if conversation_history is None:
        conversation_history = []

    # Retrieve relevant documents
    relevant_documents = retrieve_relevant_documents(
        query, n_results=n_results, threshold=threshold
    )

    logging.info("-" * 100)
    logging.info("Relevant documents: \n")
    for doc in relevant_documents:
        logging.info(doc)
        logging.info("-" * 100)
    logging.info("")

    logging.info("User's question:")
    logging.info(query)
    logging.info("")
    logging.info("-" * 100)
    logging.info("")
    
    # Build input with optional conversation history
    input_data = f"Relevant documents:\n\n{relevant_documents}\n\n"
    
    if use_memory and conversation_history:
        history_context = format_conversation_history(conversation_history)
        input_data += history_context
    
    input_data += f"Current User's question:\n\n{query}"

    # Build the RAG assistant prompt
    rag_assistant_prompt_text = build_prompt_from_config(
        prompt_config, input_data=input_data
    )

    logging.info(f"RAG assistant prompt: {rag_assistant_prompt_text}")
    logging.info("")

    # Get response from LLM
    response = llm.invoke(rag_assistant_prompt_text)
    return response.content


def is_greeting_or_chitchat(query: str) -> tuple[bool, str]:
    """
    Check if the query is a greeting or casual chat.
    Returns (is_chitchat, response)
    """
    query_lower = query.lower().strip()
    
    # Greetings
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 
                 'greetings', 'howdy', 'sup', 'what\'s up', 'yo']
    
    for greeting in greetings:
        if query_lower == greeting or query_lower.startswith(greeting + ' '):
            return True, "Hello! 👋 I'm the VIZO RAG Assistant. I'm here to help you with any questions about VIZO products and services. What would you like to know?"
    
    # Thank you responses
    thanks = ['thank you', 'thanks', 'thx', 'appreciate it', 'thank u']
    for thank in thanks:
        if thank in query_lower:
            return True, "You're welcome! 😊 Feel free to ask if you have any more questions about VIZO products."
    
    # How are you
    if any(phrase in query_lower for phrase in ['how are you', 'how r u', 'how are u']):
        return True, "I'm doing great, thank you for asking! 😊 I'm here and ready to help you with any VIZO product questions. What can I assist you with?"
    
    # Goodbye
    if any(word in query_lower for word in ['bye', 'goodbye', 'see you', 'later', 'farewell']):
        return True, "Goodbye! 👋 Feel free to come back anytime you have questions about VIZO products. Have a great day!"
    
    return False, ""


# API endpoint to ask a question
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        logging.info("=" * 80)
        logging.info("Received /ask request")
        
        data = request.get_json()
        logging.info(f"Request data: {data}")
        
        if not data or 'question' not in data:
            logging.error("No question in request")
            return jsonify({'error': 'Question is required'}), 400
        
        query = data['question']
        logging.info(f"Question: {query}")
        
        session_id = data.get('session_id', 'default')
        use_memory = data.get('use_memory', True)
        n_results = data.get('n_results', vectordb_params.get('n_results', 5))
        threshold = data.get('threshold', vectordb_params.get('threshold', 0.3))
        
        # Check if it's a greeting or chitchat
        is_chitchat, chitchat_response = is_greeting_or_chitchat(query)
        if is_chitchat:
            logging.info("Detected as chitchat, returning quick response")
            # Add to conversation history if memory is enabled
            if use_memory:
                if session_id not in conversation_histories:
                    conversation_histories[session_id] = []
                conversation_histories[session_id].append((query, chitchat_response))
            
            return jsonify({
                'question': query,
                'answer': chitchat_response,
                'session_id': session_id,
                'history_count': len(conversation_histories.get(session_id, []))
            }), 200
        
        # AUTO-INGESTION: Check for new files before processing query
        logging.info("Checking for new documents to ingest...")
        ingestion_stats = check_and_ingest_new_files()
        
        if ingestion_stats['new_files'] > 0 or ingestion_stats['modified_files'] > 0:
            logging.info(f"Ingested {ingestion_stats['new_files']} new and {ingestion_stats['modified_files']} modified files")
        
        # Get or create conversation history for this session
        if session_id not in conversation_histories:
            conversation_histories[session_id] = []
        
        conversation_history = conversation_histories[session_id]
        
        # Get response
        response = respond_to_query(
            prompt_config=rag_assistant_prompt,
            query=query,
            llm=llm,
            conversation_history=conversation_history if use_memory else [],
            use_memory=use_memory,
            n_results=n_results,
            threshold=threshold,
        )
        
        # Add to conversation history
        if use_memory:
            conversation_history.append((query, response))
            
            # Keep only the most recent pairs (max 10)
            max_history_pairs = 10
            if len(conversation_history) > max_history_pairs:
                conversation_histories[session_id] = conversation_history[-max_history_pairs:]
        
        logging.info("-" * 100)
        logging.info("LLM response:")
        logging.info(response + "\n\n")
        
        result = {
            'question': query,
            'answer': response,
            'session_id': session_id,
            'history_count': len(conversation_histories[session_id]),
            'ingestion_stats': ingestion_stats  # Include ingestion info in response
        }
        
        logging.info(f"Returning response: {result}")
        return jsonify(result), 200
        
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# Get conversation history for a session
@app.route('/history/<session_id>', methods=['GET'])
def get_history(session_id):
    if session_id not in conversation_histories:
        return jsonify({'history': [], 'count': 0}), 200
    
    history = conversation_histories[session_id]
    formatted_history = [
        {'question': q, 'answer': a} for q, a in history
    ]
    
    return jsonify({
        'session_id': session_id,
        'history': formatted_history,
        'count': len(history)
    }), 200


# Clear conversation history for a session
@app.route('/history/<session_id>', methods=['DELETE'])
def clear_history(session_id):
    if session_id in conversation_histories:
        del conversation_histories[session_id]
        return jsonify({'message': f'History cleared for session {session_id}'}), 200
    
    return jsonify({'message': 'Session not found'}), 404


# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'VIZO RAG Assistant'}), 200


# Manual ingestion endpoint (optional - trigger ingestion manually)
@app.route('/ingest', methods=['POST'])
def manual_ingest():
    """Manually trigger file ingestion."""
    try:
        stats = check_and_ingest_new_files()
        return jsonify({
            'message': 'Ingestion completed',
            'stats': stats
        }), 200
    except Exception as e:
        logging.error(f"Error during manual ingestion: {e}")
        return jsonify({'error': str(e)}), 500


# Get ingestion status
@app.route('/ingest/status', methods=['GET'])
def ingestion_status():
    """Get current ingestion status and tracked files."""
    try:
        cache = load_ingestion_cache()
        files = get_supported_files(WATCH_FOLDER)
        
        return jsonify({
            'watch_folder': WATCH_FOLDER,
            'total_files_in_folder': len(files),
            'total_tracked_files': len(cache),
            'tracked_files': [
                {
                    'file': k,
                    'ingested_at': v.get('ingested_at'),
                    'filepath': v.get('filepath')
                }
                for k, v in cache.items()
            ]
        }), 200
    except Exception as e:
        logging.error(f"Error getting ingestion status: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print(" VIZO RAG ASSISTANT API SERVER ")
    print("=" * 80)
    print("\nEndpoints:")
    print("  POST   /ask              - Ask a question")
    print("  GET    /history/<id>     - Get conversation history")
    print("  DELETE /history/<id>     - Clear conversation history")
    print("  GET    /health           - Health check")
    print("\n" + "=" * 80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)