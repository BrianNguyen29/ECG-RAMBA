import torch
import os
import glob
from typing import Dict, Any

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

class ModelLoader:
    _instances: Dict[str, Any] = {}

    @classmethod
    def get_available_models(cls):
        model_files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
        return [os.path.basename(f) for f in model_files]

    @classmethod
    def load_model(cls, model_name: str):
        if model_name in cls._instances:
            return cls._instances[model_name]

        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
             # Fallback for dev/demo if file missing
             print(f"Warning: Model {model_name} not found. Using mock inference.")
             return None

        try:
            model = torch.load(model_path)
            model.eval()
            cls._instances[model_name] = model
            return model
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None

    # Medical Knowledge Base
    KNOWLEDGE_BASE = {
        "Normal": {
            "explanation": "The heart is beating in a regular rhythm with normal electrical activity.",
            "recommendation": "Maintain a healthy lifestyle with regular exercise and a balanced diet. No immediate action required."
        },
        "Atrial Fibrillation": {
            "explanation": "An irregular and often very rapid heart rhythm (arrhythmia) that can lead to blood clots in the heart.",
            "recommendation": "Consult a cardiologist immediately. May require anticoagulants or rate control medication to prevent stroke."
        },
        "Myocardial Infarction": {
            "explanation": "Commonly known as a heart attack, this occurs when blood flow decreases or stops to a part of the heart, causing damage to the heart muscle.",
            "recommendation": "EMERGENCY: Seek immediate medical attention. This is a critical condition requiring rapid intervention."
        },
        "Arrhythmia": {
            "explanation": "A problem with the rate or rhythm of your heartbeat. The heart may beat too fast, too slow, or irregularly.",
            "recommendation": "Schedule a follow-up with a specialist for Holter monitoring to characterize the specific type of arrhythmia."
        },
        "Other": {
            "explanation": "An undefined signal pattern was detected that deviates from normal sinus rhythm.",
            "recommendation": "Clinical correlation is required. Please review the raw ECG signal manually."
        }
    }

    @classmethod
    def get_insights(cls, diagnosis):
        # Default to 'Other' if diagnosis not found exactly
        key = diagnosis if diagnosis in cls.KNOWLEDGE_BASE else "Other"
        return cls.KNOWLEDGE_BASE.get(key, cls.KNOWLEDGE_BASE["Other"])

    @classmethod
    def predict(cls, model_name: str, processed_signal):
        model = cls.load_model(model_name)
        
        result = {}

        # MOCK INFERENCE if model is missing
        if model is None:
            import random
            classes = ["Normal", "Atrial Fibrillation", "Myocardial Infarction", "Arrhythmia", "Other"]
            diagnosis = random.choice(classes)
            result = {
                "diagnosis": diagnosis,
                "confidence": round(random.uniform(0.7, 0.99), 2),
                "model_used": model_name + " (MOCK)"
            }

        else:
            # Real Inference
            try:
                tensor_input = torch.tensor(processed_signal, dtype=torch.float32).unsqueeze(0) # Add batch dim
                with torch.no_grad():
                    output = model(tensor_input)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    # Assuming simple class mapping for now - user needs to define class map
                    top_prob, top_class = torch.max(probabilities, 1)
                    
                    result = {
                        "diagnosis": f"Class {top_class.item()}", # Placeholder class name
                        "confidence": round(top_prob.item(), 2),
                        "model_used": model_name
                    }
            except Exception as e:
                return {"error": str(e)}

        # Enrich with Insights
        insights = cls.get_insights(result.get("diagnosis", "Other"))
        result.update(insights)
        
        return result

model_loader = ModelLoader()
