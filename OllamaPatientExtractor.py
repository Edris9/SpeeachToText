import csv
import os
import requests
import json
import logging
from datetime import datetime
from SchemaManager import get_schema
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class PatientExtractor:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.booking_complete = False
        self.patient_info = {
            "namn": None,
            "ålder": None,
            "adress": None,
            "besvär": None,
            "datum": "imorgon",
            "tid": "14:00"
        }
        self.csv_file = "patient_bokningar.csv"
        self._ensure_csv_exists()
    
    def _ensure_csv_exists(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Bokningsdatum", "Namn", "Ålder", "Adress", 
                               "Besvär", "Klassificering", "Besöksdatum", "Besökstid"])
    
    def extract_from_text(self, text: str):
        """Extrahera patientdata med Ollama"""
        prompt = f"""Extrahera information från texten. Svara ENDAST med JSON.

        Text: "{text}"

        JSON format:
        {{
        "namn": "för- och efternamn eller null",
        "ålder": "nummer eller null",
        "adress": "gatuadress eller null",
        "besvär": "symtom/problem eller null",
        "datum": "datum som '10 oktober' eller 'imorgon' eller null"
        }}

        JSON:"""

        payload = {
            "model": "llama3.1:8b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 200}
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", 
                                    json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()["response"].strip()
                start = result.find('{')
                end = result.rfind('}') + 1
                if start != -1 and end > start:
                    data = json.loads(result[start:end])
                    
                    for key, value in data.items():
                        if value and value != "null":
                            self.patient_info[key] = value
        except Exception as e:
            logger.error(f"Extraction error: {e}")
    
    def is_complete(self) -> bool:
        """Kolla om grundinfo finns"""
        return all([self.patient_info["namn"], 
                   self.patient_info["ålder"],
                   self.patient_info["adress"], 
                   self.patient_info["besvär"]])
    
    def classify_complaint(self) -> str:
        """Klassificera besvär"""
        complaint = (self.patient_info["besvär"] or "").lower()
        
        if any(w in complaint for w in ["ont", "smärta", "ben", "arm", "hand"]):
            return "Ortopedi"
        elif any(w in complaint for w in ["feber", "hosta", "snuva"]):
            return "Infektion"
        elif any(w in complaint for w in ["utslag", "klåda"]):
            return "Hud"
        return "Övrigt"
    
    def save_to_csv(self) -> bool:
        if not self.is_complete():
            return False
        
        
        
        from SchemaManager import get_schema
        schema = get_schema()
        
        datum = schema.format_datum(self.patient_info["datum"] or "imorgon")
        available = schema.get_available_slots(datum)
        
        if not available:
            return False
        
        tid = available[0]
        if not schema.book_slot(datum, tid):
            return False
        
        self.patient_info["datum"] = datum
        self.patient_info["tid"] = tid
        
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.patient_info["namn"],
                    self.patient_info["ålder"],
                    self.patient_info["adress"],
                    self.patient_info["besvär"],
                    self.classify_complaint(),
                    datum,
                    tid
                ])
            self.last_booking = {"datum": datum, "tid": tid}
            self.booking_complete = True  
            print(f"✓ Bokad: {datum} kl {tid}")
            return True
        except:
            return False
    
    def reset(self):
        """Rensa för ny patient"""
        self.booking_complete = False  # Reset state
        self.patient_info = {
            "namn": None,
            "ålder": None,
            "adress": None,
            "besvär": None,
            "datum": None,  # Ändra till None
            "tid": None     # Ändra till None
        }
    
    def get_info(self):
        return self.patient_info.copy()


# Singleton
_extractor = None

def get_extractor() -> PatientExtractor:
    global _extractor
    if _extractor is None:
        _extractor = PatientExtractor()
    return _extractor

def extract_and_save() -> bool:
    """Spara patient om komplett"""
    extractor = get_extractor()
    return extractor.save_to_csv()