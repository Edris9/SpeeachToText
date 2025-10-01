from cgitb import text
import requests
import json
import logging
from typing import Optional
from datetime import datetime
from OllamaPatientExtractor import get_extractor, extract_and_save
# Konfigurera logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class OllamaResponder:
    def __init__(self, model_name: str = "llama3.1:8b", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"
        self.conversation_history = []
        self.extractor = get_extractor()  # Lägg till denna
        
        if not self._test_connection():
            logger.warning(f"Kunde inte ansluta till Ollama på {ollama_url}")
    
    def _test_connection(self) -> bool:
        """Testa om Ollama körs och modellen finns"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                return self.model_name in available_models
            return False
        except Exception as e:
            logger.error(f"Fel vid anslutning till Ollama: {e}")
            return False
    
    def _get_current_context(self) -> str:
        """Hämta aktuell kontext"""
        now = datetime.now()
        weekdays = ["måndag", "tisdag", "onsdag", "torsdag", "fredag", "lördag", "söndag"]
        months = ["januari", "februari", "mars", "april", "maj", "juni",
                 "juli", "augusti", "september", "oktober", "november", "december"]
        
        weekday = weekdays[now.weekday()]
        month = months[now.month - 1]
        
        return f"""Aktuell information:
        - Datum: {now.day} {month} {now.year}
        - Dag: {weekday}
        - Tid: {now.strftime('%H:%M')}"""
            
    def _build_prompt(self, user_input: str) -> str:
        info = self.extractor.get_info()
        
        saknas = []
        if not info["namn"]: saknas.append("namn")
        if not info["ålder"]: saknas.append("ålder")
        if not info["adress"]: saknas.append("adress")
        if not info["besvär"]: saknas.append("besvär")
        
        if not saknas:
            return f"Perfekt! Jag har all information. Bokningen är klar."
        
        return f"""Du bokar tid för 1177.

    HAR REDAN: {', '.join([k for k,v in info.items() if v])}
    SAKNAS: {', '.join(saknas)}

    Fråga om FÖRSTA saknade. Kort fråga, max 10 ord.

    Användare: {user_input}
    Du:"""
    
    def generate_response(self, user_input: str) -> Optional[str]:
        if not user_input or not user_input.strip():
            return None
        
        # Extrahera data från input
        self._auto_extract(user_input)
        
        # Kolla om vi har allt
        if self.extractor.is_complete():
            return "Perfekt! Din bokning är klar. Du kommer till läkare för dina besvär. Något mer?"
        
        prompt = self._build_prompt(user_input)
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 50,
                "num_ctx": 2048,  # Lägg till denna
            }
            }
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get("response", "").strip().strip('"\'')
                
                if ai_response:
                    self._update_history(user_input, ai_response)
                    return ai_response
                else:
                    return "Förlåt, jag kunde inte generera ett svar just nu."
            else:
                logger.error(f"Ollama API fel: {response.status_code}")
                return "Det uppstod ett tekniskt problem."
                
        except requests.RequestException as e:
            logger.error(f"Nätverksfel: {e}")
            return "Jag kan inte svara just nu på grund av anslutningsproblem."
        except Exception as e:
            logger.error(f"Oväntat fel: {e}")
            return "Ett oväntat fel uppstod."


    def _update_history(self, user_input: str, ai_response: str):
        """Uppdatera konversationshistorik"""
        self.conversation_history.append(f"Användare: {user_input}")
        self.conversation_history.append(f"Assistent: {ai_response}")
        
        # Öka från 20 till 40 för att behålla mer kontext
        if len(self.conversation_history) > 40:
            self.conversation_history = self.conversation_history[-40:]

    def _auto_extract(self, text: str):
        self.extractor.extract_from_text(text)
    
    def clear_history(self):
        """Rensa konversationshistorik"""
        self.conversation_history.clear()

# Singleton instans
_responder_instance = None

def get_responder(model_name: str = "llama3.1:8b") -> OllamaResponder:
    """Hämta eller skapa responder instans"""
    global _responder_instance
    if _responder_instance is None:
        _responder_instance = OllamaResponder(model_name)
    return _responder_instance

def generate_ai_response(user_text: str, model_name: str = "llama3.1:8b") -> str:
    """Generera AI-svar"""
    if not user_text or not user_text.strip():
        return "Jag hörde inget tydligt. Kan du upprepa?"
    
    responder = get_responder(model_name)
    response = responder.generate_response(user_text)
    return response if response else "Förlåt, jag kunde inte svara på det."

def clear_conversation():
    """Rensa konversationshistorik"""
    global _responder_instance
    if _responder_instance:
        _responder_instance.clear_history()


