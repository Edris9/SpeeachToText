import pandas as pd
import ollama
from datetime import datetime

# Läs CSV
df = pd.read_csv(
    r"D:\Python\Python Programmeri ng\own project\1177_simulated\SpeeachToText\schema_oktober_november_2025.csv",
    sep=';'
)

def extrahera_datum(fråga):
    """Försök hitta datum i frågan"""
    ord = fråga.lower().split()
    for i, ord_text in enumerate(ord):
        if ord_text in ['oktober', 'november']:
            # Leta efter siffra före månadsnamnet
            for j in range(max(0, i-3), i):
                try:
                    dag = int(ord[j].strip('.:!?'))
                    månad = '10' if ord_text == 'oktober' else '11'
                    return f"2025-{månad}-{dag:02d}"
                except:
                    continue
    return None

def chatta_om_schema(fråga):
    # Hitta relevant datum
    datum = extrahera_datum(fråga)
    
    if datum:
        # Filtrera endast relevant dag
        relevant_data = df[df['Datum'] == datum]
        data_text = relevant_data.to_string(index=False)
    else:
        # Visa sammanfattning
        data_text = df.to_string(index=False, max_rows=50)
    
    prompt = f"""Schema för oktober-november 2025.

{data_text}

Fråga: {fråga}
Svara kort och tydligt på svenska."""

    response = ollama.chat(
        model='llama3.1:8b-instruct-q8_0',
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    return response['message']['content']

while True:
    fråga = input("\nFråga: ")
    if fråga.lower() == 'exit':
        break
    print(f"Svar: {chatta_om_schema(fråga)}\n")