import requests
import json
import logging
from typing import Optional

# Konfigurera logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class OllamaSpellChecker:
    def __init__(self, model_name: str = "llama3.1:8b", ollama_url: str = "http://localhost:11434"):
        """
        Initialisera spell checker med Ollama
        
        Args:
            model_name: Namnet på Ollama-modellen att använda
            ollama_url: URL till Ollama API
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"
        
        # Testa anslutning
        if not self._test_connection():
            logger.warning(f"Kunde inte ansluta till Ollama på {ollama_url}")
    
    def _test_connection(self) -> bool:
        """Testa om Ollama körs och modellen finns"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                if self.model_name in available_models:
                    return True
                else:
                    logger.error(f"Modell {self.model_name} finns inte. Tillgängliga: {available_models}")
            return False
        except Exception as e:
            logger.error(f"Fel vid anslutning till Ollama: {e}")
            return False
    
    def correct_text(self, text: str) -> Optional[str]:
        """
        Korrigera stavning och grammatik i texten
        
        Args:
            text: Text att korrigera
            
        Returns:
            Korrigerad text eller None om fel uppstod
        """
        if not text or not text.strip():
            return text
        
        # Prompt för att korrigera svenska text
        prompt = f"""Du är en svensk språkkorrigerare. Din uppgift är att korrigera stavfel, grammatikfel och göra texten mer naturlig på svenska.

VIKTIGT: 
- Svara ENDAST med den korrigerade texten
- Ingen förklaring eller kommentar
- Behåll samma betydelse och stil
- Om texten redan är korrekt, returnera den oförändrad

Text att korrigera: "{text}"

Korrigerad text:"""

        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Låg temperatur för konsistens
                    "top_p": 0.9,
                    "num_predict": 200,  # Begränsa output-längd
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
                corrected_text = result.get("response", "").strip()
                
                # Rensa eventuella citattecken eller extra formatering
                corrected_text = corrected_text.strip('"\'')
                
                # Om korrigeringen är tom eller för lång, returnera original
                if not corrected_text or len(corrected_text) > len(text) * 3:
                    return text
                
                return corrected_text
            else:
                logger.error(f"Ollama API fel: {response.status_code}")
                return text
                
        except requests.RequestException as e:
            logger.error(f"Nätverksfel vid korrigering: {e}")
            return text
        except Exception as e:
            logger.error(f"Oväntat fel vid korrigering: {e}")
            return text
    
    def correct_with_fallback(self, text: str) -> str:
        """
        Korrigera text med fallback till original vid fel
        
        Args:
            text: Text att korrigera
            
        Returns:
            Korrigerad text eller original text om korrigering misslyckas
        """
        corrected = self.correct_text(text)
        return corrected if corrected is not None else text


# Singleton instans för enkel användning
_spell_checker_instance = None

def get_spell_checker(model_name: str = "llama3.1:8b") -> OllamaSpellChecker:
    """
    Hämta eller skapa spell checker instans
    
    Args:
        model_name: Ollama modell att använda
        
    Returns:
        OllamaSpellChecker instans
    """
    global _spell_checker_instance
    if _spell_checker_instance is None:
        _spell_checker_instance = OllamaSpellChecker(model_name)
    return _spell_checker_instance

def correct_transcription(text: str, model_name: str = "llama3.1:8b") -> str:
    """
    Enkel funktion för att korrigera transkriberad text
    
    Args:
        text: Text att korrigera
        model_name: Ollama modell att använda
        
    Returns:
        Korrigerad text
    """
    if not text or not text.strip():
        return text
    
    checker = get_spell_checker(model_name)
    return checker.correct_with_fallback(text)


