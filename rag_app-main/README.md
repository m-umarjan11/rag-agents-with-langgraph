# RAG Application

A Retrieval-Augmented Generation (RAG) application built with FastAPI, LangGraph, Google Gemini, and Pinecone vector database.

## Features

- **PDF Document Processing**: Upload and process PDF documents for knowledge base
- **Intelligent Retrieval**: Vector-based similarity search using embeddings
- **LangGraph Agent**: Sophisticated agent workflow for query processing
- **Pinecone Vector Database**: Scalable, cloud-based vector storage
- **Session Management**: Maintain conversation history and context
- **RESTful API**: FastAPI-based endpoints for chat and file management
- **Google Gemini Integration**: Using Gemini for embeddings and text generation

## Architecture

```
rag_app/
│
├── main.py                 # FastAPI entrypoint
├── api/                    # API routes
│   ├── routes_chat.py      # Chat endpoints
│   ├── routes_files.py     # File management endpoints
├── core/                   # Core configurations
│   ├── config.py           # Environment configuration
├── services/               # Business logic
│   ├── data_ingestion_service.py      # PDF processing and chunking
│   ├── embeddings_service.py          # Gemini embeddings
│   ├── vectordb_service.py           # Vector database operations
├── utils/                  # Utility functions
│   ├── logger.py           # Centralized logging
├── langgraph_agent/        # LangGraph agent
│   ├── agent.py            # Main agent logic
│   ├── tools.py            # Agent tools
│   ├── state.py            # State management
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Google AI API key
- Pinecone account and API key

### Installation

1. **Clone and navigate to the project directory:**
   ```bash
   cd rag_app
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:
   ```env
   # Required
   GOOGLE_API_KEY=your_google_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=us-east-1-aws  # or your preferred region
   PINECONE_INDEX_NAME=rag-index
   ```

### Running the Application

1. **Start the RAG application:**
   ```bash
   python main.py
   ```

The application will automatically:
- Connect to your Pinecone account using the API key
- Create the index if it doesn't exist (with 768 dimensions for Gemini embeddings)
- Start the FastAPI server

The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- Interactive API docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

## API Endpoints

### Chat Endpoints

- `POST /api/v1/chat` - Send a chat message
- `GET /api/v1/chat/sessions/{session_id}` - Get chat history
- `DELETE /api/v1/chat/sessions/{session_id}` - Clear chat session

### File Management Endpoints

- `POST /api/v1/files/upload` - Upload and process PDF file
- `GET /api/v1/files` - List all uploaded files
- `GET /api/v1/files/{file_id}` - Get file information
- `PUT /api/v1/files/{file_id}` - Update existing file
- `DELETE /api/v1/files/{file_id}` - Delete file and its embeddings

## Usage Examples

### Upload a PDF Document

```bash
curl -X POST "http://localhost:8000/api/v1/files/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "metadata=Research paper on AI"
```

### Chat with the Knowledge Base

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the main topic of the uploaded document?",
    "session_id": "my-session-123"
  }'
```

## Configuration

Key configuration options in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `PINECONE_API_KEY` | Your Pinecone API key | Required |
| `PINECONE_ENVIRONMENT` | Pinecone region (e.g., us-east-1-aws) | Required |
| `PINECONE_INDEX_NAME` | Name of your Pinecone index | `rag-index` |
| `CHUNK_SIZE` | Text chunk size for document processing | `1000` |
| `CHUNK_OVERLAP` | Overlap between text chunks | `200` |
| `MAX_FILE_SIZE` | Maximum upload file size in bytes | `52428800` (50MB) |
| `LOG_LEVEL` | Logging level | `INFO` |

## Development

### Running Tests

```bash
pytest
```

### Code Structure

- **Services**: Business logic for data processing, embeddings, and vector operations
- **API Routes**: FastAPI endpoints for external interface
- **LangGraph Agent**: Sophisticated workflow for query processing
- **Configuration**: Centralized environment-based configuration
- **Utilities**: Shared utilities like logging

### Adding New Features

1. **New Tools**: Add to `langgraph_agent/tools.py`
2. **New Endpoints**: Add to appropriate file in `api/`
3. **New Services**: Add to `services/` directory
4. **Configuration**: Update `core/config.py` and `.env.example`

## Troubleshooting

### Common Issues

1. **Pinecone Connection Issues**
   - Verify your API key is correct and active
   - Ensure the environment/region is valid (e.g., us-east-1-aws)
   - Check that your Pinecone account has sufficient quota
   - Verify the index name doesn't contain invalid characters

2. **Index Creation Issues**
   - Ensure your Pinecone plan supports the required features
   - Check if you've reached your index limit
   - Verify the region supports serverless indexes

3. **PDF Processing Issues**
   - Ensure PyPDF2 is installed
   - Check file size limits (50MB default)
   - Verify PDF is not corrupted or password-protected

4. **Gemini API Issues**
   - Verify Google API key is correct and active
   - Check rate limits and quota
   - Ensure internet connectivity

5. **Embedding Dimension Issues**
   - The application uses 768-dimensional embeddings (Gemini default)
   - If changing embedding models, ensure dimension compatibility

### Logs

Check application logs for detailed error information. Logs include:
- Document processing status
- Pinecone operations (upsert, query, delete)
- Index statistics and health
- API request/response details
- Agent workflow steps

### Getting Help

For Pinecone-specific issues:
- Check the [Pinecone Documentation](https://docs.pinecone.io/)
- Monitor your [Pinecone Console](https://app.pinecone.io/) for usage and limits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.