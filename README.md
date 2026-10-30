# RAG-PDF-SummQA

A local Retrieval-Augmented Generation (RAG) application for **question answering** and **automatic summarization** of scientific PDF documents.

The project combines PDF text extraction, text chunking, semantic embeddings, FAISS vector search, and locally runnable Hugging Face language models. It was developed around scientific papers concerning wastewater treatment plants and explores how semantic retrieval can help users find relevant information in research documents.

## Features

- Extracts text from PDF documents.
- Splits extracted text into overlapping chunks for retrieval.
- Creates semantic embeddings and stores them in a FAISS vector index.
- Retrieves relevant passages in response to a natural-language question.
- Generates answers using a locally runnable sequence-to-sequence language model.
- Produces summaries of indexed PDF documents.
- Provides a command-line interface for choosing between summarization and question answering.
- Runs inference locally after the required models have been downloaded; no paid OpenAI API is required.

## How it works

The application has two main workflows:

### 1. Document indexing

1. PDF files are read from the `pdfs/` directory.
2. Text is extracted and cleaned.
3. The text is divided into chunks (configured with a chunk size of 1,000 characters and an overlap of 200 characters).
4. A sentence-transformer model converts the chunks into semantic embeddings.
5. The embeddings and associated document chunks are stored in a FAISS index for similarity search.

### 2. Question answering

1. The user enters a question through the command-line interface.
2. The vector store retrieves passages that are semantically relevant to the question.
3. The retrieved context is prepared for the language model.
4. A locally runnable FLAN-T5 model generates an answer based on the retrieved information.

### 3. Summarization

1. The application obtains the indexed document text.
2. The text is divided into manageable sections.
3. A summarization model generates summaries for the sections.
4. The partial summaries are combined and processed to produce a consolidated summary.

## Architecture

```text
Scientific PDF files
        |
        v
   PDF extraction
        |
        v
   Text cleaning
        |
        v
  Text chunking
        |
        v
 Sentence embeddings
        |
        v
    FAISS index
        |
        +------------------------------+
        |                              |
        v                              v
 Question-answering workflow     Summarization workflow
        |                              |
 Retrieve relevant chunks        Collect document text
        |                              |
 Prepare context                 Split into sections
        |                              |
 FLAN-T5 answer generation       DistilBART summarization
        |                              |
        v                              v
   Generated answer             Generated summary
```

### Main files

| File | Responsibility |
|---|---|
| `main.py` | Application entry point, model initialization, and command-line interface. |
| `vector_store.py` | PDF indexing, embedding generation, FAISS persistence, and similarity search. |
| `query_handler.py` | Question-answering and document-summarization workflows. |
| `utils.py` | Text extraction, cleaning, normalization, and supporting utilities. |
| `requirements.txt` | Python dependencies. |

## Models and technologies

| Component | Model / technology | Purpose |
|---|---|---|
| Summarization | `sshleifer/distilbart-cnn-12-6` | Generates summaries of document text. |
| Question answering | `google/flan-t5-large` | Generates answers from retrieved context. |
| Embeddings | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | Converts text chunks into semantic vectors. |
| Vector search | FAISS | Stores and retrieves vector embeddings. |
| PDF processing | PyPDF2 / LangChain PDF utilities | Extracts text from PDF files. |
| ML framework | PyTorch and Hugging Face Transformers | Loads and runs the language models. |
| Application language | Python | Implements the application and workflows. |

## Project structure

```text
RAG-PDF-SummQA/
├── pdfs/
│   └── .gitkeep
├── main.py
├── query_handler.py
├── vector_store.py
├── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

PDF documents and generated vector-store files are not included in the repository. Add your own PDFs to the `pdfs/` directory.

## Installation

### Requirements

- Python 3.12
- pip
- An internet connection for the initial download of the Hugging Face models
- Sufficient system memory and disk space for the models and dependencies

The application can run on CPU. A compatible CUDA-enabled environment may also be used if configured appropriately.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/RAG-PDF-SummQA.git
cd RAG-PDF-SummQA
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Add PDF documents

Place the scientific PDF files you want to process inside the `pdfs/` directory.

The repository does not distribute research papers. Make sure you have the right to use any documents you add.

## Usage

Start the application from the project root:

```bash
python main.py
```

On the first run, the application may need to download the embedding and language models from Hugging Face. It then processes the PDFs in the `pdfs/` directory and builds or updates the local vector index.

Use the command-line options when prompted:

- `summarize` — generate a summary of the indexed PDF documents.
- `question` — ask a question about the indexed documents.
- `exit` — close the application.

For question answering, enter your question when prompted. The response is generated using the retrieved document context.

## Configuration

Some retrieval and text-processing parameters are configured in the source code. The current project configuration includes:

- Chunk size: `1,000` characters
- Chunk overlap: `200` characters
- Question-answering retrieval: up to `15` relevant chunks

Review `vector_store.py` and `query_handler.py` if you want to adjust indexing, retrieval, or text-processing behavior.

## Notes

- The application is designed for local document processing and local model inference after the initial model downloads.
- Model loading and inference can require substantial memory, particularly when using the larger FLAN-T5 model.
- The generated FAISS index is local application data and is intentionally excluded from version control.
- Generated answers and summaries should be checked against the original research papers, especially when used for academic or scientific work.

## Project status

This is an experimental academic/personal project exploring RAG, semantic search, question answering, and document summarization. Runtime behavior may vary depending on the machine, installed dependencies, and available memory.

## License

No license has been added yet. Unless a license is included, the repository remains under the default copyright terms; others do not automatically receive permission to reuse or redistribute the code.
