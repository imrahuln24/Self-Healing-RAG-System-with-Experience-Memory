# Self-Healing-RAG-System-with-Experience-Memory

A sophisticated Retrieval-Augmented Generation (RAG) system that learns from past retrieval experiences to improve answer quality over time. Built with LangGraph, LangChain, and OpenAI.

## Features

- **Self-Improving**: Uses an evaluator to score retrieval quality and iteratively improves answers
- **Experience Memory**: Stores successful retrieval strategies and reuses them for similar questions
- **Multi-Iteration**: Automatically retries with feedback until quality threshold is met
- **Document Search**: Searches PDF documents using semantic similarity
- **Tool-Based Architecture**: Uses LangGraph for structured agent workflows

## Architecture

The system uses a multi-node graph architecture:

1. **Retriever Agent**: LLM that decides which tools to use
2. **Tools**: 
   - `retriever_tool`: Searches the PDF document
   - `memory_search`: Searches past successful experiences
3. **Evaluator**: Scores answers 0-10 based on relevance, completeness, and ability to answer
4. **Experience Storage**: Saves successful retrieval strategies for future use

The agent follows this flow:
1. Search memory first for similar past questions
2. Search the document if needed
3. Evaluate the answer quality
4. If score < 8, retry with feedback (max 5 iterations)
5. If score >= 8, store the successful experience

## Requirements

- Python 3.8+
- OpenAI API key
- PDF document (default: `Stock_Market_Performance_2024.pdf`)

## Installation

1. Install dependencies:
```bash
pip install langgraph langchain langchain-openai langchain-community langchain-chroma pypdf
```

2. Set your OpenAI API key:
```bash
set OPENAI_API_KEY=your_api_key_here
```

3. Place your PDF document in the same directory as the script (or update the `pdf_path` variable)

## Usage

Run the script:
```bash
python "Self Healing RAG System with Experience Memory.py"
```

The system will start an interactive prompt where you can ask questions about your PDF document.

Example:
```
What is your question: How did the stock market perform in Q1 2024?
=== ANSWER ===
[AI response based on document retrieval]
```

Type `exit` or `quit` to stop the agent.

## Configuration

Key parameters you can adjust:

- `THRESHOLD`: Minimum score to accept an answer (default: 8)
- `MAX_ITERATIONS`: Maximum retry attempts (default: 5)
- `chunk_size`: Document chunk size for vector store (default: 1000)
- `search_kwargs["k"]`: Number of chunks to retrieve (default: 5)
- `persist_directory`: Where vector stores are saved

## How It Works

### Document Processing
1. Loads PDF using PyPDFLoader
2. Splits documents into chunks using RecursiveCharacterTextSplitter
3. Creates embeddings with OpenAI's text-embedding-3-small
4. Stores in ChromaDB for semantic search

### Experience Memory
- Stores successful question-answer pairs
- Includes the best retrieval query used
- Saves evaluator feedback and scores
- Enables learning from past successes

### Self-Healing Loop
1. Agent retrieves information using tools
2. Evaluator scores the answer (0-10)
3. If score < threshold, agent retries with feedback
4. Loop continues until threshold met or max iterations reached
5. Successful experiences are stored for future use

## Models Used

- **Main LLM**: GPT-4o-mini (temperature=0 for deterministic output)
- **Evaluator LLM**: GPT-4o (temperature=0 for consistent evaluation)
- **Embeddings**: text-embedding-3-small

## File Structure

```
.
├── Self Healing RAG System with Experience Memory.py  # Main script
├── Stock_Market_Performance_2024.pdf                 # Source document
└── README.md                                         # This file
```

Vector stores are persisted to the directory specified in `persist_directory`.

## Limitations

- Requires OpenAI API key with credits
- PDF must be in the same directory (or path updated)
- Memory is stored locally - not persistent across different machines
- Evaluator scores are subjective and may vary

## Future Improvements

- Add support for multiple document types
- Implement persistent memory storage (database)
- Add web interface
- Support for custom evaluation criteria
- Add analytics dashboard for performance tracking

## License

This project is provided as-is for educational and research purposes.
