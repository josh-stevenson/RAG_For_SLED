import os
import requests
import json

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- Custom Embeddings Class for Nutanix Granite Embedding API ---
# This class needs to be defined here as well because ChromaDB requires
# the same embedding function to load a persistent database.
class NutanixGraniteEmbeddings(Embeddings):
    def __init__(self, api_key: str, endpoint_url: str, model_name: str):
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.session = requests.Session()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._get_embeddings([text])[0]

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float"
        }

        response = None
        try:
            response = self.session.post(self.endpoint_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            if "data" in result and len(result["data"]) > 0:
                embeddings = [item["embedding"] for item in result["data"]]
                return embeddings
            else:
                raise ValueError("API response did not contain expected 'data' or embeddings.")

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if response is not None:
                print(f"Response status: {response.status_code}")
                print(f"Response text: {response.text}")
            raise
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from API response: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during embedding: {e}")
            raise

# --- RAG System Core Functions ---

# Configuration (needs to be consistent with ingest_data.py and environment variables)
CHROMA_DB_DIR = "my_vector_db"
EMBEDDING_API_MODEL_NAME = "embedcpu"
NUTANIX_GRANITE_EMBEDDING_ENDPOINT = "https://ai.nutanix.com/api/v1/embeddings"
LLM_API_MODEL_NAME = "llama3370b"
NUTANIX_LLM_ENDPOINT = "https://ai.nutanix.com/api/v1"

# Global variables to store initialized components
_rag_chain = None
_vectorstore = None

def _initialize_rag_components():
    """Initializes and returns the RAG chain and vectorstore."""
    global _rag_chain, _vectorstore

    if _rag_chain is not None: # If already initialized, return existing
        return _rag_chain, _vectorstore

    # Ensure API Key is set
    NUTANIX_API_KEY = os.getenv("NUTANIX_API_KEY")
    if not NUTANIX_API_KEY:
        raise ValueError("NUTANIX_API_KEY environment variable not set. Please set it.")

    # 1. Load the Persistent ChromaDB
    print(f"\n--- Loading ChromaDB from '{CHROMA_DB_DIR}' ---")
    embeddings_model = NutanixGraniteEmbeddings(
        api_key=NUTANIX_API_KEY,
        endpoint_url=NUTANIX_GRANITE_EMBEDDING_ENDPOINT,
        model_name=EMBEDDING_API_MODEL_NAME
    )
    # Ensure the database exists before trying to load it
    if not os.path.exists(CHROMA_DB_DIR):
        raise FileNotFoundError(f"ChromaDB directory '{CHROMA_DB_DIR}' not found. Please run ingest_data.py first.")
    _vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings_model)
    print("ChromaDB loaded successfully.")

    # 2. Initialize the Remote LLM (Nutanix AI Endpoint)
    print(f"\n--- Initializing Remote Language Model (Nutanix AI Endpoint: '{LLM_API_MODEL_NAME}') ---")
    llm = ChatOpenAI(
        model=LLM_API_MODEL_NAME,
        temperature=0.0,
        openai_api_key=NUTANIX_API_KEY,
        base_url=NUTANIX_LLM_ENDPOINT
    )
    print(f"Remote LLM '{llm.model_name}' initialized successfully.")

    # 3. Create a Retriever from the Vector Database
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. Build the RAG Chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the provided context only. Respond concisely and in English.\nContext: {context}"),
        ("user", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)
    _rag_chain = create_retrieval_chain(retriever, document_chain)
    print("RAG chain built successfully.")
    
    return _rag_chain, _vectorstore

def ask_rag_system(query: str) -> dict:
    """
    Asks the RAG system a question and returns the LLM's answer and retrieved context.
    Components are initialized on first call.
    """
    rag_chain, _ = _initialize_rag_components()
    response = rag_chain.invoke({"input": query})
    return response

# Example usage (for testing this module directly)
if __name__ == "__main__":
    # Ensure NUTANIX_API_KEY is set in your environment before running this directly
    print("Running rag_core.py as main for testing...")
    # This will initialize components and ask a dummy question
    try:
        test_query = "What is the main topic of these documents?"
        print(f"\nTesting with query: '{test_query}'")
        response = ask_rag_system(test_query)
        print("\nTest LLM Answer:")
        print(response["answer"])
        print("\nTest Retrieved Context (Documents used by LLM):")
        if response["context"]:
            for j, doc in enumerate(response["context"]):
                print(f"  - Document {j+1} (Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')})")
                print(f"    Content (first 100 chars): {doc.page_content[:100]}...")
        else:
            print("  No relevant documents were retrieved for the test query.")
    except Exception as e:
        print(f"Error during rag_core test: {e}")
