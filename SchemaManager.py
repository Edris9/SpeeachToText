import csv
import os
from datetime import datetime
from datetime import datetime, timedelta
import re

class SchemaManager:
    def __init__(self, csv_file="schema_oktober_november_2025.csv"):
        self.csv_file = csv_file
    
    def get_available_slots(self, datum: str) -> list:
        """Hämta lediga tider för ett datum"""
        slots = []
        
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                if row['Datum'] == datum and row['Status'] == 'Ledigt':
                    slots.append(row['Tid'])
        
        return slots
    
    def book_slot(self, datum: str, tid: str) -> bool:
        """Boka en tid"""
        rows = []
        booked = False
        
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                if row['Datum'] == datum and row['Tid'] == tid and row['Status'] == 'Ledigt':
                    row['Status'] = 'Bokad'
                    booked = True
                rows.append(row)
        
        if booked:
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['Datum', 'Tid', 'Status'], delimiter=';')
                writer.writeheader()
                writer.writerows(rows)
        
        return booked
    
    def format_datum(self, user_datum: str) -> str:
        """Konvertera 'imorgon', 'idag', '9 oktober' till YYYY-MM-DD"""
        if not user_datum:
            return (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        today = datetime.now()
        
        if 'imorgon' in user_datum.lower():
            return (today + timedelta(days=1)).strftime('%Y-%m-%d')
        elif 'idag' in user_datum.lower():
            return today.strftime('%Y-%m-%d')
        
        # "9 oktober"
        match = re.search(r'(\d+)\s+(januari|februari|mars|april|maj|juni|juli|augusti|september|oktober|november|december)', user_datum.lower())
        if match:
            dag = int(match.group(1))
            månader = {"januari":1, "februari":2, "mars":3, "april":4, "maj":5, "juni":6,
                      "juli":7, "augusti":8, "september":9, "oktober":10, "november":11, "december":12}
            månad = månader[match.group(2)]
            return f"{today.year}-{månad:02d}-{dag:02d}"
        
        return user_datum

# Singleton
_schema = None

def get_schema() -> SchemaManager:
    global _schema
    if _schema is None:
        _schema = SchemaManager()
    return _schema