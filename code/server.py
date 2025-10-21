from flask import Flask, request, jsonify
from datetime import datetime
import os
import logging
from dotenv import load_dotenv  # type: ignore
from utils import load_yaml_config
from prompt_builder import build_prompt_from_config
from langchain_google_genai import ChatGoogleGenerativeAI # pyright: ignore[reportMissingImports]
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage # pyright: ignore[reportMissingImports]
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR
from vector_db_ingest import get_db_collection, embed_documents
from flask_cors import CORS



app = Flask(__name__)

# Add after: app = Flask(__name__)
CORS(app)

logger = logging.getLogger()


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

# Initialize collection
collection = get_db_collection(collection_name="Vizo_Product_Manual")

# Conversation memory (in production, use sessions or database)
conversation_histories = {}


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


# API endpoint to ask a question
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({'error': 'Question is required'}), 400
        
        query = data['question']
        session_id = data.get('session_id', 'default')
        use_memory = data.get('use_memory', True)
        n_results = data.get('n_results', vectordb_params.get('n_results', 5))
        threshold = data.get('threshold', vectordb_params.get('threshold', 0.3))
        
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
        
        return jsonify({
            'question': query,
            'answer': response,
            'session_id': session_id,
            'history_count': len(conversation_histories[session_id])
        }), 200
        
    except Exception as e:
        logging.error(f"Error processing query: {e}")
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