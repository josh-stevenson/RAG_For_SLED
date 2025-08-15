RAG System with Nutanix AI Endpoints
This project implements a Retrieval Augmented Generation (RAG) system designed to answer questions based on a corpus of PDF documents. It leverages Nutanix AI inference endpoints for both text embeddings and large language model (LLM) inference, providing a robust and scalable solution for document-based Q&A.

Features
PDF Ingestion: Automatically loads and processes PDF documents from a specified directory.

Intelligent Chunking: Splits documents into manageable chunks to optimize embedding quality and retrieval accuracy.

Vector Database: Uses ChromaDB to store document embeddings, allowing for efficient semantic search.

Nutanix AI Integration: Seamlessly integrates with Nutanix AI endpoints for:

Embeddings: Using the NutanixGraniteEmbeddings model.

LLM Inference: Using a remote chat completion model (e.g., llama-3-3-70b).

Interactive Chatbot: A user-friendly chat interface built with Streamlit for real-time, document-based Q&A.

Modular Design: The project is structured into logical components (ingest_data.py, rag_core.py, streamlit_app.py) for clarity and maintainability.

Prerequisites
Before you begin, ensure you have the following installed on your system:

Python 3.10+

Anaconda or Miniconda: Highly recommended for managing project dependencies.

Nutanix AI Account & API Key: A key is required to access the embedding and LLM endpoints.

Project Setup
1. Environment Configuration
It's best practice to use a dedicated Conda environment.

First, create a file named environment.yml with the following content:

# environment.yml
name: rag_nutanix_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - requests
  - chromadb
  - streamlit
  - pip:
    - pdfplumber==0.7.0
    - langchain
    - langchain-community
    - langchain-openai


Next, open your terminal (or Anaconda Prompt) and run the following commands to create and activate the environment:

# Navigate to your project directory
cd /path/to/your/project

# Create the Conda environment
conda env create -f environment.yml

# Activate the environment
conda activate rag_nutanix_env


2. API Key Configuration
For security, your Nutanix AI API key should be set as an environment variable.

On Windows (Command Prompt):

set NUTANIX_API_KEY=YOUR_ACTUAL_NUTANIX_API_KEY


On macOS/Linux:

export NUTANIX_API_KEY=YOUR_ACTUAL_NUTANIX_API_KEY


IMPORTANT: Replace YOUR_ACTUAL_NUTANIX_API_KEY with your real API key.

Workflow Guide
Step 1: Ingest Data
Place your PDF documents inside a folder named data/pdfs/ in the project root. Then, run the ingest_data.py script to process the documents and build the vector database.

python ingest_data.py


This script will load the PDFs, create text chunks, generate embeddings using the Nutanix API, and save them to a my_vector_db directory.

Step 2: Run the Chatbot
Once the vector database is created, you can launch the interactive chatbot application using Streamlit.

streamlit run streamlit_app.py


This command will open a web browser window with the chatbot interface. You can then ask questions about the content of your PDF documents.

File Structure
The project is organized as follows:

.
├── data/
│   └── pdfs/             # Place your PDF documents here
├── my_vector_db/         # (Generated) ChromaDB vector store
├── environment.yml       # Conda environment definition
├── ingest_data.py        # Script for data ingestion and vector DB creation
├── rag_core.py           # Core RAG logic and API interaction
├── streamlit_app.py      # Streamlit chatbot application
└── README.md             # This file

