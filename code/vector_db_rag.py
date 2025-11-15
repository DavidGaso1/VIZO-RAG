import os
import logging
from dotenv import load_dotenv
from utils import load_yaml_config
from prompt_builder import build_prompt_from_config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR
from vector_db_ingest import get_db_collection, embed_documents

logger = logging.getLogger()


def setup_logging():
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(os.path.join(OUTPUTS_DIR, "rag_assistant_improved.log"))
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


load_dotenv()

# To avoid tokenizer parallelism warning from huggingface
os.environ["TOKENIZERS_PARALLELISM"] = "false"

collection = get_db_collection(collection_name="Vizo_Product_Manual")


def retrieve_relevant_documents(
    query: str,
    n_results: int = 15,  # Increased from 5
    threshold: float = 0.5,  # Relaxed from 0.3 (lower distance = more similar)
) -> list[str]:
    """
    Query the ChromaDB database with a string query.

    Args:
        query (str): The search query string
        n_results (int): Number of results to return (default: 15)
        threshold (float): Threshold for the cosine distance score (default: 0.5)

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

    logging.info(f"Found {len(relevant_results['documents'])} relevant documents after filtering")
    
    # Log distances to help tune threshold
    if relevant_results["distances"]:
        logging.info(f"Distance range: {min(relevant_results['distances']):.3f} to {max(relevant_results['distances']):.3f}")

    return relevant_results["documents"]


def retrieve_with_query_expansion(
    query: str,
    n_results: int = 15,
    threshold: float = 0.5,
) -> list[str]:
    """
    Retrieve documents using query expansion for broad questions.
    
    For questions like "all products", we search multiple times with different
    query formulations to ensure comprehensive coverage.
    """
    # Check if this is a broad "list all" type query
    # Remove punctuation and normalize query
    normalized_query = query.lower().replace('?', '').replace('!', '').replace('.', '').strip()
    
    broad_keywords = ["all products", "all vizo products", "list products", "what products", 
                      "available products", "all services", "list services", "what services", 
                      "everything", "complete list", "tell me about all", "what are all",
                      "show me all", "give me all"]
    
    is_broad_query = any(keyword in normalized_query for keyword in broad_keywords)
    
    if is_broad_query:
        logging.info("Detected broad query - using query expansion")
        
        # Generate multiple search queries to cover all products
        expanded_queries = [
            query,  # Original query
            "ViZO products and services overview",
            "digital wallet cryptocurrency gold investment",
            "bill payment airtime data utility",
            "money transfer bank smart account",
            "gift cards services features",
        ]
        
        all_documents = []
        seen_docs = set()
        
        for expanded_query in expanded_queries:
            docs = retrieve_relevant_documents(expanded_query, n_results=10, threshold=threshold)
            for doc in docs:
                # Avoid duplicates
                if doc not in seen_docs:
                    all_documents.append(doc)
                    seen_docs.add(doc)
        
        logging.info(f"Retrieved {len(all_documents)} unique documents using query expansion")
        return all_documents
    else:
        # For specific queries, use standard retrieval
        return retrieve_relevant_documents(query, n_results=n_results, threshold=threshold)


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
    n_results: int = 15,  # Increased default
    threshold: float = 0.5,  # Relaxed default
    use_memory: bool = True,
    use_query_expansion: bool = True,  # New parameter
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
        use_query_expansion: Whether to use query expansion for broad queries
    
    Returns:
        Response string
    """
    if conversation_history is None:
        conversation_history = []

    # Retrieve relevant documents with optional query expansion
    if use_query_expansion:
        relevant_documents = retrieve_with_query_expansion(
            query, n_results=n_results, threshold=threshold
        )
    else:
        relevant_documents = retrieve_relevant_documents(
            query, n_results=n_results, threshold=threshold
        )

    logging.info("-" * 100)
    logging.info(f"Retrieved {len(relevant_documents)} relevant documents\n")
    for i, doc in enumerate(relevant_documents, 1):
        logging.info(f"Document {i}:")
        logging.info(doc[:200] + "..." if len(doc) > 200 else doc)
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
    rag_assistant_prompt = build_prompt_from_config(
        prompt_config, input_data=input_data
    )

    logging.info(f"RAG assistant prompt length: {len(rag_assistant_prompt)} characters")
    logging.info("")

    # Get response from LLM
    response = llm.invoke(rag_assistant_prompt)
    return response.content


def print_conversation_summary(conversation_history: list):
    """Print a summary of the conversation."""
    if not conversation_history:
        print("\n No conversation history yet.\n")
        return
    
    print("\n" + "=" * 80)
    print(f" CONVERSATION SUMMARY ({len(conversation_history)} Q&A pairs)")
    print("=" * 80)
    for i, (question, answer) in enumerate(conversation_history, 1):
        print(f"\nQ{i}: {question}")
        print(f"A{i}: {answer[:150]}{'...' if len(answer) > 150 else ''}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    setup_logging()
    app_config = load_yaml_config(APP_CONFIG_FPATH)
    prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)

    # Use enhanced RAG prompt
    rag_assistant_prompt = prompt_config["rag_assistant_prompt_enhanced"]

    vectordb_params = app_config["vectordb"]
    llm_model = app_config["llm"]
    
    # Initialize LLM once
    llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.7)
    
    # Conversation memory settings
    use_memory = True
    use_query_expansion = True  # Enable query expansion
    max_history_pairs = 3
    conversation_history = []

    print("\n" + "=" * 80)
    print(" VIZO RAG ASSISTANT WITH IMPROVED RETRIEVAL")
    print("=" * 80)
    print("\nFeatures:")
    print("  ✓ Query expansion for comprehensive answers")
    print("  ✓ Increased retrieval capacity (up to 15+ documents)")
    print("  ✓ Conversation memory")
    print("\nCommands:")
    print("  - Type your question to get an answer")
    print("  - 'config' - Change retrieval parameters")
    print("  - 'memory on/off' - Toggle conversation memory")
    print("  - 'expansion on/off' - Toggle query expansion")
    print("  - 'history' - View conversation history")
    print("  - 'clear' - Clear conversation history")
    print("  - 'exit' - Quit the application")
    print(f"\n Memory: {'ON' if use_memory else 'OFF'} | Query Expansion: {'ON' if use_query_expansion else 'OFF'}")
    print("=" * 80 + "\n")

    exit_app = False
    while not exit_app:
        query = input("Enter a question or command: ").strip()
        
        if not query:
            continue
            
        if query.lower() == "exit":
            print("\n Goodbye! Thanks for using VIZO RAG Assistant.\n")
            exit_app = True
            exit()

        elif query.lower() == "config":
            print("\n  Configuration Settings:")
            threshold = float(input(f"Enter the retrieval threshold (current: {vectordb_params['threshold']}): ") or vectordb_params['threshold'])
            n_results = int(input(f"Enter the Top K value (current: {vectordb_params['n_results']}): ") or vectordb_params['n_results'])
            max_history_pairs = int(input(f"Enter max conversation history pairs (current: {max_history_pairs}): ") or max_history_pairs)
            
            vectordb_params = {
                "threshold": threshold,
                "n_results": n_results,
            }
            print(f" Configuration updated!\n")
            continue
        
        elif query.lower() == "memory on":
            use_memory = True
            print(" Conversation memory turned ON\n")
            continue
        
        elif query.lower() == "memory off":
            use_memory = False
            print(" Conversation memory turned OFF\n")
            continue
        
        elif query.lower() == "expansion on":
            use_query_expansion = True
            print(" Query expansion turned ON\n")
            continue
        
        elif query.lower() == "expansion off":
            use_query_expansion = False
            print(" Query expansion turned OFF\n")
            continue
        
        elif query.lower() == "history":
            print_conversation_summary(conversation_history)
            continue
        
        elif query.lower() == "clear":
            conversation_history = []
            print(" Conversation history cleared!\n")
            continue

        # Process the query
        try:
            response = respond_to_query(
                prompt_config=rag_assistant_prompt,
                query=query,
                llm=llm,
                conversation_history=conversation_history if use_memory else [],
                use_memory=use_memory,
                use_query_expansion=use_query_expansion,
                **vectordb_params,
            )
            
            # Add to conversation history
            conversation_history.append((query, response))
            
            # Keep only the most recent pairs
            if len(conversation_history) > max_history_pairs:
                conversation_history = conversation_history[-max_history_pairs:]
            
            # Display response
            logging.info("-" * 100)
            logging.info("LLM response:")
            logging.info(response + "\n\n")
            
            print("\n" + "=" * 80)
            print(" ASSISTANT RESPONSE:")
            print("=" * 80)
            print(f"\n{response}\n")
            print("=" * 80)
            print(f" History: {len(conversation_history)}/{max_history_pairs} Q&A pairs | Memory: {'ON' if use_memory else 'OFF'} | Expansion: {'ON' if use_query_expansion else 'OFF'}")
            print("=" * 80 + "\n")
            
        except Exception as e:
            print(f"\n Error: {e}\n")
            logging.error(f"Error processing query: {e}")
