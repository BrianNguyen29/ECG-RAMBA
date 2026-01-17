import os
import sys
from app.services.rag_engine import RAGEngine

# Fix for Windows SQLite issues if needed
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    """Simple chunking by character count (overlap could be added)."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def main():
    print("🚀 Starting Knowledge Ingestion...")
    
    # Initialize Engine
    try:
        rag = RAGEngine()
    except Exception as e:
        print(f"❌ Failed to init RAGEngine: {e}")
        return

    base_dir = "data/knowledge"
    domains = ["cardio", "neuro", "physio"]
    
    total_chunks = 0
    
    for domain in domains:
        domain_path = os.path.join(base_dir, domain)
        if not os.path.exists(domain_path):
            continue
            
        print(f"\n📂 Processing domain: {domain}")
        for filename in os.listdir(domain_path):
            if not (filename.endswith(".txt") or filename.endswith(".md")):
                continue
                
            filepath = os.path.join(domain_path, filename)
            print(f"   📄 Reading {filename}...", end="")
            
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            chunks = chunk_text(content)
            print(f" ({len(chunks)} chunks)")
            
            for i, chunk in enumerate(chunks):
                success = rag.ingest(
                    text=chunk,
                    metadata={"source": filename, "chunk_id": i},
                    domain=domain
                )
                if success:
                    total_chunks += 1
                else:
                    print(f"      ⚠️ Failed to ingest chunk {i}")

    print(f"\n✅ Ingestion Complete! Total chunks indexed: {total_chunks}")
    
    # Verify Retrieval
    print("\n🔍 Verifying Retrieval (Query: 'ECG interpretation')...")
    result = rag.retrieve("ECG interpretation", domain="cardio")
    if result:
        print(f"   Outcome: Found {len(result)} chars of context.")
    else:
        print("   Outcome: ⚠️ No context found.")

if __name__ == "__main__":
    main()
