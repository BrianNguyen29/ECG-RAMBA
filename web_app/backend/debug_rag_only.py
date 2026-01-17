import time
try:
    print("DEBUG: Importing RAGEngine...")
    t0 = time.time()
    from app.services.rag_engine import RAGEngine
    print(f"DEBUG: Import took {time.time() - t0:.4f}s")

    print("DEBUG: Initializing RAGEngine...")
    t1 = time.time()
    rag = RAGEngine()
    print(f"DEBUG: Init took {time.time() - t1:.4f}s")

    query = "What is the frequency of Alpha waves?"
    domain = "neuro"
    
    print(f"DEBUG: Retrieving '{query}' from '{domain}'...")
    context = rag.retrieve(query, domain=domain)
    
    print(f"DEBUG: Result length: {len(context)}")
    print(f"DEBUG: Content peek: {context[:100]}...")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
