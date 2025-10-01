import torch
import torchaudio
import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import warnings
import logging

# Importera våra egna moduler

from Respons import generate_ai_response

# Stäng av varningar
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings("ignore", message="You have passed task=transcribe")

# Konfig
CONFIG = {
    "MODEL_NAME": "openai/whisper-medium",
    "SAMPLE_RATE": 16000,
    "SILENCE_THRESHOLD": 0.008,
    "MAX_DURATION": 30,  # Ändra från 12 till 30 sekunder
    "SILENCE_AFTER_SPEECH": 2.5,
    "INITIAL_SILENCE_TIMEOUT": 8,
    "EXIT_SILENCE_TIMEOUT": 10,
    "CHUNK_DURATION": 0.2,
    "LANGUAGE": "swedish",
    "MAX_TRANSCRIPTION_LENGTH": 50,
}

def initialize_model():
    """Ladda och konfigurera Whisper-modell"""
    print("Laddar Whisper-modell...")
    
    processor = WhisperProcessor.from_pretrained(CONFIG["MODEL_NAME"])
    model = WhisperForConditionalGeneration.from_pretrained(CONFIG["MODEL_NAME"])
    
    model.config.pad_token_id = model.config.eos_token_id
    model.config.forced_decoder_ids = None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    if torch.__version__ >= "2.0" and device == "cuda":
        model = torch.compile(model)
    
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    return processor, model, device

def record_audio():
    """Spela in ljud med tystnadsdetektion"""
    chunk_size = int(CONFIG["SAMPLE_RATE"] * CONFIG["CHUNK_DURATION"])
    audio_chunks = []
    speech_detected = False
    silence_after_speech = 0
    # total_silence = 0  # Ta bort denna rad
    
    print("AI lyssnar...")
    
    stream = sd.InputStream(
        samplerate=CONFIG["SAMPLE_RATE"],
        channels=1,
        dtype="float32",
        blocksize=chunk_size
    )
    
    with stream:
        for _ in range(int(CONFIG["MAX_DURATION"] / CONFIG["CHUNK_DURATION"])):
            chunk, _ = stream.read(chunk_size)
            chunk = np.squeeze(chunk)
            
            volume = np.max(np.abs(chunk))
            
            if volume > CONFIG["SILENCE_THRESHOLD"]:
                speech_detected = True
                silence_after_speech = 0
                audio_chunks.append(chunk)
                # total_silence = 0  # Ta bort denna rad
            else:
                if speech_detected:
                    audio_chunks.append(chunk)
                    silence_after_speech += CONFIG["CHUNK_DURATION"]
                    if silence_after_speech > CONFIG["SILENCE_AFTER_SPEECH"]:
                        break
           
    
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        max_samples = CONFIG["SAMPLE_RATE"] * CONFIG["MAX_DURATION"]
        if len(full_audio) > max_samples:
            full_audio = full_audio[:max_samples]
        return full_audio
    return None

def transcribe_audio(audio_data, processor, model, device):
    """Transkribera ljud till text"""
    if audio_data is None or isinstance(audio_data, str):
        return audio_data
    
    try:
        audio_tensor = torch.from_numpy(audio_data.astype(np.float32)).to(device)
        
        audio_tensor = torchaudio.transforms.Resample(
            orig_freq=CONFIG["SAMPLE_RATE"], 
            new_freq=CONFIG["SAMPLE_RATE"]
        )(audio_tensor)
        
        audio_tensor = torchaudio.functional.lowpass_biquad(
            audio_tensor, 
            CONFIG["SAMPLE_RATE"], 
            cutoff_freq=3000
        )
        
        audio_tensor = audio_tensor / (torch.max(torch.abs(audio_tensor)) + 1e-6)
        
        inputs = processor(
            audio_tensor.cpu().numpy(),
            sampling_rate=CONFIG["SAMPLE_RATE"],
            return_tensors="pt",
            return_attention_mask=True
        )
        
        input_features = inputs.input_features.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad(), torch.amp.autocast(device_type=device):
            predicted_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                language=CONFIG["LANGUAGE"],
                task="transcribe",
                max_new_tokens=CONFIG["MAX_TRANSCRIPTION_LENGTH"],
                num_beams=1,
            )
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        if len(transcription.split()) > CONFIG["MAX_TRANSCRIPTION_LENGTH"] // 2:
            transcription = " ".join(transcription.split()[:CONFIG["MAX_TRANSCRIPTION_LENGTH"] // 2])
        
        return transcription.strip() if transcription else None
        
    except Exception as e:
        print(f"Fel vid transkribering: {e}")
        return None

def process_text(original_text):
    if not original_text or not original_text.strip():
        return
    
    print(f"Användare: {original_text}")
    ai_response = generate_ai_response(original_text)
    print(f"AI: {ai_response}")
    
    # Kolla om vi har grundinfo (namn, ålder, adress, besvär)
    from OllamaPatientExtractor import get_extractor
    extractor = get_extractor()
    info = extractor.get_info()
    
    # Visa vad vi har
    if info["namn"] and info["ålder"] and info["adress"] and info["besvär"]:
        print("\n✅ GRUNDINFO INSAMLAD:")
        print(f"  Namn: {info['namn']}")
        print(f"  Ålder: {info['ålder']}")
        print(f"  Adress: {info['adress']}")
        print(f"  Besvär: {info['besvär']}")
        
        # Fyll i standardvärden för datum/tid om de saknas
        if not info["datum"]:
            extractor.update_patient_info("datum", "imorgon")
        if not info["tid"]:
            extractor.update_patient_info("tid", "14:00")
        
        from OllamaPatientExtractor import extract_and_save
        if extract_and_save():
            print("✅ SPARAT TILL CSV!\n")
    
    print()

def main():
    """Huvudfunktion"""
    processor, model, device = initialize_model()
    print("AI Röstassistent redo!")
    print("=" * 40)
    
    try:
        while True:
            audio = record_audio()
            
            if isinstance(audio, str) and audio == "EXIT":
                print("Avslutar...")
                break
            
            if audio is not None:
                text = transcribe_audio(audio, processor, model, device)
                if text and text.strip():
                    process_text(text)
                    
    except KeyboardInterrupt:
        print("\nAvslutar...")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()