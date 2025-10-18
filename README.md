# VIZO RAG Assistant

A Retrieval-Augmented Generation (RAG) system built with LangChain and ChromaDB for intelligent document querying and response generation.

## Features

- 🔍 Vector-based semantic search using ChromaDB
- 🤖 LLM integration with Gemini 2.5
- 💡 Configurable reasoning strategies (Chain of Thought, ReAct, Self-Ask)
- 🧠 Conversation memory with configurable strategies
- 📝 YAML-based configuration for prompts and system settings
- 📊 Comprehensive logging system

## Project Structure

```
.
├── code/
│   ├── config/
│   │   ├── config.yaml        # Main configuration
│   │   └── prompt_config.yaml # Prompt templates
│   ├── vector_db_ingest.py   # Document ingestion
│   ├── vector_db_rag.py      # RAG implementation
│   ├── prompt_builder.py     # Prompt construction
│   └── utils.py              # Utility functions
├── data/
│   └── vizo_product_manual.md # Source documents
└── outputs/
    └── vector_db/            # ChromaDB storage
```

## Prerequisites

- Python 3.11+
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DavidGaso1/VIZO-RAG
cd vizo-rag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r code/requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Configuration

The system is configured through two YAML files:

- `code/config/config.yaml`: System settings including:
  - LLM model selection
  - Vector database parameters
  - Memory strategy settings
  - Reasoning strategy templates

- `code/config/prompt_config.yaml`: Prompt engineering templates

## Usage

1. Ingest documents into the vector database:
```bash
python code/vector_db_ingest.py
```

2. Start the RAG assistant:
```bash
python code/vector_db_rag.py
```

3. Enter queries when prompted. Special commands:
   - Type 'config' to adjust retrieval parameters
   - Type 'exit' to quit

## Key Components

### Vector Database (ChromaDB)
- Document storage and retrieval
- Semantic search capabilities
- Configurable similarity thresholds

### RAG System
- Retrieves relevant context from the vector database
- Generates responses using selected LLM
- Maintains conversation history
- Implements various reasoning strategies

### Memory Management
- Configurable conversation history
- Trimming and summarization strategies
- Contextual response generation

## Dependencies

Main packages:
- langchain (0.3.25+)
- chromadb (1.0.12+)
- sentence-transformers (4.1.0+)
- python-dotenv (1.1.0+)
- pyyaml (6.0.2+)

For a complete list, see `code/requirements.txt`

### Adding New Features
1. Update configuration in `config.yaml`
2. Add new functionality in relevant modules
3. Update tests as needed
4. Document changes in this README

## Logging

Logs are stored in `outputs/rag_assistant.log` and include:
- Retrieved documents
- User queries
- LLM responses
- System operations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit a pull request

## License

