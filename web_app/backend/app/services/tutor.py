import os
import google.generativeai as genai
from typing import List, Dict, Optional, Any
import json

# =============================================================================
# LAYER 4: KNOWLEDGE & MEMORY
# =============================================================================
class KnowledgeBase:
    def __init__(self):
        # Initialize RAG Engine
        from app.services.rag_engine import RAGEngine
        self.rag = RAGEngine()
    
    def retrieve(self, query: str = "", domain: str = "cardio") -> str:
        """
        Retrieves relevant guidelines using Vector Search (ChromaDB).
        Defaults to 'cardio' domain.
        """
        # If query is empty (e.g. init), fetch generic context or summary
        search_query = query if query and len(query) > 5 else "ECG interpretation guidelines"
        
        context = self.rag.retrieve(search_query, domain=domain)
        
        if not context:
            return "Standard ACC/AHA Guidelines apply. (RAG returned no specific context)"
            
        print(f"RAG Retrieved {len(context)} chars for query: '{search_query}'")
        return context

# =============================================================================
# LAYER 2: AGENT MODULES (Smart Solver)
# =============================================================================
class ReasoningEngine:
    def __init__(self, api_key: str):
        if api_key:
            genai.configure(api_key=api_key)
            # Use gemini-2.0-flash-exp (Lighter/Faster)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp', 
                generation_config={"response_mime_type": "application/json"})
        else:
            self.model = None

    async def solve(self, student_msg: str, context: Dict, history: List[Dict], knowledge: str) -> Dict:
        """
        Executes the 'Analysis Loop' and 'Solve Loop' from the Architecture Diagram.
        Returns a structured JSON response.
        """
        if not self.model:
            # Re-initialize if model name changed or first run
            try:

                # Use Gemini 2.0 Flash (Experimental) for speed/lightweight
                model_name = "gemini-2.0-flash-exp" 
                self.model = genai.GenerativeModel(model_name, 
                    generation_config={"response_mime_type": "application/json"})
            except Exception as e:
                print(f"Error init model: {e}")
                return {"reply": f"AI Init Error: {e}", "action": None}
        
        if not self.model:
             return {"reply": "Error: AI not configured.", "action": None}

        # Context Formatting
        ai_data = f"""
        - Diagnosis: {context.get('diagnosis', 'N/A')}
        - Heart Rate: {context.get('hr', 'N/A')} bpm
        - Reliability: {context.get('confidence', 'N/A')}
        """

        # Chain-of-Thought Prompt for the "Smart Solver"
        prompt = f"""
        ACT AS: "DeepBioSignal Tutor", an expert clinical reasoning agent for ECG and EEG.
        
        GOAL: Guide the medical student using the Socratic Method for clinical cases, but provide DIRECT, CLEAR DEFINITIONS for factual questions. You support Cardiology (ECG) and Neurology (EEG).
        
        INPUT LAYERS:
        1. STUDENT QUERY: "{student_msg}"
        2. PATIENT DATA: {ai_data}
        3. KNOWLEDGE BASE: {knowledge}

        TASK (Layer 2 - Smart Solver):
        1. ANALYZE: What is the student's misconception or knowledge gap?
        2. PLAN: Determine the pedagogical strategy (e.g., "Hint about P-waves" for ECG, "Ask about Alpha rhythm" for EEG).
        3. QUIZ: If the student asks for a quiz, generate a relevant Multiple Choice Question with 4 options.
        4. ACT: Generate the response text. Rely heavily on the KNOWLEDGE BASE provided.
        5. TOOL: Decide if a visual action is needed (e.g., highlight a lead).

        OUTPUT STRUCTURE (Layer 1 - Response):
        Return ONLY valid JSON:
        {{
            "thought_process": "Analysis of student state...",
            "pedagogical_plan": "Strategy used...",
            "reply": "The actual text response to the student...",
            "action": {{ "type": "highlight_lead", "value": "V1" }} OR null
        }}
        """

        # Convert history (Simplified for context window)
        # We perform a stateless inference optimized for speed here
        try:
            response = await self.model.generate_content_async(prompt)
            text = response.text.strip()
            # Clean markdown code blocks if present
            if text.startswith("```"):
                text = text.strip("`").replace("json", "").strip()
            return json.loads(text)
        except Exception as e:
            print(f"Reasoning Error: {e}")
            return {
                "reply": "I analyzed the case but encountered a format error. Let's discuss the ECG features directly.", 
                "action": None
            }

# =============================================================================
# LAYER 1: ORCHESTRATOR
# =============================================================================
class TutorService:
    def __init__(self):
        self.kb = KnowledgeBase()
        self.brain = None
        
        # Lazy load settings to avoid circular import issues if possible, 
        # or just import at top. Better to use the global settings object.
        from app.core.config import settings
        
        # Use settings.GEMINI_API_KEY which is loaded from .env
        api_key = settings.GEMINI_API_KEY
        
        if api_key:
            self.brain = ReasoningEngine(api_key)
        else:
            print("WARNING: GEMINI_API_KEY missing in Settings.")

    async def generate_reply(self, message: str, context: Optional[Dict] = None, history: List[Dict] = []) -> str:
        """
        Orchestrates the flow: Input -> Knowledge Retreival -> Reasoning -> Output.
        """
        import time
        start_time = time.time()
        
        if not self.brain:
            return "System Error: Gemini Key missing."

        # 1. Layer 3: Retrieve Knowledge
        t0 = time.time()
        import asyncio
        
        # Determine domain from context
        domain = "cardio" # Default
        if context:
            view = context.get("current_view", "")
            modality = context.get("modality", "")
            if view == "research" or modality == "EEG":
                domain = "neuro"
        
        # Offload blocking RAG retrieval to a separate thread
        knowledge = await asyncio.to_thread(self.kb.retrieve, message, domain=domain)
        t1 = time.time()
        print(f"PERF: KB Retrieval took {t1 - t0:.4f}s")

        # 2. Layer 2: Smart Solver (Reasoning)
        t2 = time.time()
        result = await self.brain.solve(message, context or {}, history, knowledge)
        t3 = time.time()
        print(f"PERF: Gemini API Call took {t3 - t2:.4f}s")
        
        # 3. Layer 1: Guided Learning (Visual + Text)
        reply_text = result.get("reply", "Error generating reply.")
        action = result.get("action")
        
        # Append logic for Frontend Action Parser
        if action and action.get('type') == 'highlight_lead':
            lead = action.get('value')
            reply_text += f" <SHOW_LEAD value=\"{lead}\">"
        
        total_time = time.time() - start_time
        print(f"PERF: Total Generate Reply took {total_time:.4f}s")
        
        return reply_text

# Singleton
tutor_service = TutorService()
