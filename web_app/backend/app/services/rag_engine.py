import chromadb
import google.generativeai as genai
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    def __init__(self, persist_path: str = "data/vector_db"):
        """
        Initialize ChromaDB client.
        Data is stored locally in 'data/vector_db'.
        """
        # Ensure directory exists
        os.makedirs(persist_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_path)
        
        # Define Collections for each domain
        self.collections = {
            "cardio": self.client.get_or_create_collection("cardio_knowledge"),
            "neuro": self.client.get_or_create_collection("neuro_knowledge"),
            "physio": self.client.get_or_create_collection("physio_knowledge")
        }
        
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini API."""
        try:
            # use "models/text-embedding-004" (newer) or "models/embedding-001"
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Embedding Error: {e}")
            return []

    def ingest(self, text: str, metadata: Dict, domain: str = "cardio") -> bool:
        """Video index a chunk of text into the specified domain collection."""
        if domain not in self.collections:
            return False
            
        collection = self.collections[domain]
        embedding = self._get_embedding(text)
        
        if not embedding:
            return False
            
        # Add to Chroma
        collection.add(
            documents=[text],
            metadatas=[metadata],
            embeddings=[embedding],
            ids=[f"{domain}_{metadata.get('source', 'doc')}_{metadata.get('chunk_id', '0')}"]
        )
        return True

    def retrieve(self, query: str, domain: str = "cardio", n_results: int = 3) -> str:
        """Retrieve relevant context for a query."""
        import time
        start = time.time()
        print(f"DEBUG: RAGEngine.retrieve start for '{query[:20]}...', domain={domain}")
        
        if domain not in self.collections:
            print(f"DEBUG: Domain {domain} not found.")
            return ""
            
        collection = self.collections[domain]
        
        try:
            # Embed query
            t1 = time.time()
            query_embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )['embedding']
            print(f"DEBUG: Embedding took {time.time() - t1:.4f}s")
            
            t2 = time.time()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            print(f"DEBUG: Chroma Query took {time.time() - t2:.4f}s")
            
            # Format context
            context_parts = []
            if results['documents']:
                for doc in results['documents'][0]:
                    context_parts.append(doc)
            
            print(f"DEBUG: Total Retrieve took {time.time() - start:.4f}s")
            return "\n\n".join(context_parts)
            
        except Exception as e:
            print(f"Retrieval Error: {e}")
            return ""
