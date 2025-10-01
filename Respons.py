import requests
import json
import logging
from OllamaPatientExtractor import get_extractor

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class SimpleResponder:
    def __init__(self):
        self.extractor = get_extractor()
        self.conversation_history = []
    
    def generate_response(self, user_input: str) -> str:
        self.extractor.extract_from_text(user_input)
        info = self.extractor.get_info()

        if self.extractor.booking_complete:
            if "ändra" in user_input.lower() and "datum" in user_input.lower():
                # Nollställ datum och tid
                self.extractor.patient_info["datum"] = None
                self.extractor.patient_info["tid"] = None
                self.extractor.booking_complete = False
                
                # Extrahera nytt datum direkt från input
                self.extractor.extract_from_text(user_input)
                
                # Om datum hittades, gå vidare
                if self.extractor.patient_info["datum"]:
                    from SchemaManager import get_schema
                    schema = get_schema()
                    datum = schema.format_datum(self.extractor.patient_info["datum"])
                    slots = schema.get_available_slots(datum)
                    
                    if slots:
                        return f"Lediga tider {datum}: {', '.join(slots[:3])}. Vilken tid?"
                    else:
                        return f"Inga lediga tider {datum}. Annat datum?"
                
                return "Vilket nytt datum vill du ha?"
        
        if not info["namn"]:
            return "Vad heter du?"
        if not info["ålder"]:
            return f"Okej {info['namn']}, hur gammal är du?"
        if not info["adress"]:
            return "Var bor du?"
        if not info["besvär"]:
            return "Vad har du för besvär?"
        
        # LÄGG TILL: Fråga om datum
        if not info["datum"]:
            return "Vilket datum vill du ha tid?"
        
        # Visa lediga tider
        from SchemaManager import get_schema
        schema = get_schema()
        datum = schema.format_datum(info["datum"])
        slots = schema.get_available_slots(datum)
        
        if not slots:
            return f"Tyvärr inga lediga tider {datum}. Annat datum?"
        
        if not info["tid"]:
            return f"Lediga tider {datum}: {', '.join(slots[:3])}. Vilken tid?"
        if self.extractor.save_to_csv():
            last = getattr(self.extractor, 'last_booking', {})
            return f"Perfekt! Bokad {last.get('datum', '')} kl {last.get('tid', '')}."
        
        return "Kunde inte boka."

        
        

_responder = None

def get_responder():
    global _responder
    if _responder is None:
        _responder = SimpleResponder()
    return _responder

def generate_ai_response(user_text: str) -> str:
    responder = get_responder()
    return responder.generate_response(user_text)