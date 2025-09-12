# RAG (Retrieval-Augmented Generation) Demo Project

This project demonstrates a complete RAG system that enhances Large Language Model (LLM) responses with relevant information retrieved from your documents. It uses Gemini for generation and FAISS for efficient similarity search.

## 📚 Project Overview

The system processes documents (PDF, HTML, Markdown) through several stages:
1. **Ingestion**: Extract and clean text from documents
2. **Splitting**: Break documents into smaller, meaningful chunks
3. **Embedding**: Convert text chunks into vector representations
4. **Retrieval**: Find relevant chunks for user queries
5. **Generation**: Combine retrieved context with Gemini LLM to generate accurate answers

## 🗂️ Project Structure

- `app.py`: Main Streamlit web interface
- `ingest.py`: Document ingestion and text extraction
- `splitter.py`: Text chunking and splitting
- `embed_index.py`: Vector embedding and FAISS index creation
- `generator.py`: Core RAG functionality (retrieval + generation)
- `retriver.py`: Standalone retrieval utilities

## 📁 Directory Structure

```
.
├── ingestion_input/     # Place your input documents here
├── ing_out_split_in/    # Ingested documents output
├── split_out_emd_in/    # Split chunks output
└── emd_out_retr_in/     # FAISS index and metadata
```

## 🚀 Getting Started

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key**
   - Create a `.env` file in the project root
   - Add your Gemini API key: `GEMINI_API_KEY=your_key_here`

3. **Process Documents**
   - Place your documents in `ingestion_input/`
   - Run the pipeline:
     ```bash
     python ingest.py      # Extract text from documents
     python splitter.py    # Split into chunks
     python embed_index.py # Create FAISS index
     ```

4. **Launch the UI**
   ```bash
   streamlit run app.py
   ```

## 💡 Features

- **Multi-format Support**: Process PDF, HTML, and Markdown files
- **Smart Chunking**: Uses both sentence-based and semantic splitting
- **Efficient Search**: FAISS vector similarity search
- **Interactive UI**: User-friendly Streamlit interface
- **Source Citations**: Answers include numbered citations to source documents
- **Debug Info**: View retrieval metrics and chunks

## 🛠️ Technologies Used

- **LLM**: Google Gemini
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS
- **Text Processing**: 
  - PDFMiner.six for PDF extraction
  - BeautifulSoup4 for HTML parsing
  - html2text for Markdown processing
- **Web UI**: Streamlit

## 📋 Configuration

The project uses fixed paths and configurations for simplicity:
- Input documents: `ingestion_input/`
- Chunk size: 800 tokens
- PDF overlap: 200 tokens
- Default top-k: 5 documents
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- LLM model: `gemini-1.5-flash`

## 🔄 Processing Flow

1. `ingest.py` reads documents and extracts clean text
2. `splitter.py` divides text into optimal chunks
3. `embed_index.py` creates vector representations
4. When a query is received:
   - `generator.py` finds relevant chunks
   - Combines them with the query
   - Sends to Gemini for answer generation
   - Returns answer with citations

## 🎯 Use Cases

- Question answering over your document collection
- Information retrieval with source attribution
- Document exploration and analysis
- Fact checking with citations

## ⚠️ Limitations

- Only processes PDF, HTML, and Markdown files
- Fixed chunk sizes might not be optimal for all content
- Requires Gemini API key
- Local file storage (no database)

## 🤝 Contributing

Feel free to:
- Add support for more document types
- Implement dynamic chunk sizing
- Add more LLM options
- Improve the retrieval algorithm

## 📝 License

This project is for educational purposes. Use responsibly and ensure you have appropriate licenses for all components.