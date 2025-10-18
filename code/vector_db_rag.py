import os
import logging
from dotenv import load_dotenv  # type: ignore
from utils import load_yaml_config
from prompt_builder import build_prompt_from_config
from langchain_google_genai import ChatGoogleGenerativeAI # pyright: ignore[reportMissingImports]
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage # pyright: ignore[reportMissingImports]
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR
from vector_db_ingest import get_db_collection, embed_documents

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


load_dotenv()

# To avoid tokenizer parallelism warning from huggingface
os.environ["TOKENIZERS_PARALLELISM"] = "false"

collection = get_db_collection(collection_name="Vizo_Product_Manual")


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
    conversation_history: list = None, # type: ignore
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
    rag_assistant_prompt = build_prompt_from_config(
        prompt_config, input_data=input_data
    )

    logging.info(f"RAG assistant prompt: {rag_assistant_prompt}")
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
    print(f"CONVERSATION SUMMARY ({len(conversation_history)} Q&A pairs)")
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
    max_history_pairs = 10  # Keep last 10 Q&A pairs
    conversation_history = []

    print("\n" + "=" * 80)
    print(" VIZO RAG ASSISTANT ")
    print("=" * 80)
    print("\nCommands:")
    print("  - Type your question to get an answer")
    print("  - 'config' - Change retrieval parameters")
    print("  - 'memory on/off' - Toggle conversation memory")
    print("  - 'history' - View conversation history")
    print("  - 'clear' - Clear conversation history")
    print("  - 'exit' - Quit the application")
    print(f"\n Memory: {'ON' if use_memory else 'OFF'} | Keeping last {max_history_pairs} Q&A pairs")
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
            print(f" History: {len(conversation_history)}/{max_history_pairs} Q&A pairs | Memory: {'ON' if use_memory else 'OFF'}")
            print("=" * 80 + "\n")
            
        except Exception as e:
            print(f"\n Error: {e}\n")
            logging.error(f"Error processing query: {e}")
