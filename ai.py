import torch
import torchaudio
import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import warnings
import logging

# Stäng av varningar
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# Konfig från din kod
CONFIG = {
    "MODEL_NAME": "openai/whisper-small",
    "SAMPLE_RATE": 16000,
    "SILENCE_THRESHOLD": 0.008,
    "MAX_DURATION": 12,
    "SILENCE_AFTER_SPEECH": 1.2,
    "INITIAL_SILENCE_TIMEOUT": 5,
    "EXIT_SILENCE_TIMEOUT": 10,
    "CHUNK_DURATION": 0.2,
    "LANGUAGE": "swedish",
    "MAX_TRANSCRIPTION_LENGTH": 50,
}

# Ladda modell
processor = WhisperProcessor.from_pretrained(CONFIG["MODEL_NAME"])
model = WhisperForConditionalGeneration.from_pretrained(CONFIG["MODEL_NAME"])

# Viktig konfiguration från din kod
model.config.pad_token_id = model.config.eos_token_id
model.config.forced_decoder_ids = None

# GPU/CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Torch compile för snabbare inference (från din kod)
if torch.__version__ >= "2.0" and device == "cuda":
    model = torch.compile(model)

print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def record_with_silence_detection():
    """Spela in med tystnadsdetektion"""
    chunk_size = int(CONFIG["SAMPLE_RATE"] * CONFIG["CHUNK_DURATION"])
    audio_chunks = []
    speech_detected = False
    silence_after_speech = 0
    total_silence = 0
    printed_no_input = False
    
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
                total_silence = 0
                printed_no_input = False
            else:
                if speech_detected:
                    audio_chunks.append(chunk)
                    silence_after_speech += CONFIG["CHUNK_DURATION"]
                    if silence_after_speech > CONFIG["SILENCE_AFTER_SPEECH"]:
                        break
                else:
                    total_silence += CONFIG["CHUNK_DURATION"]
                    if total_silence >= CONFIG["INITIAL_SILENCE_TIMEOUT"] and not printed_no_input:
                        print("Inget input...")
                        printed_no_input = True
                        return None
                    if total_silence >= CONFIG["EXIT_SILENCE_TIMEOUT"]:
                        return "EXIT"
    
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        max_samples = CONFIG["SAMPLE_RATE"] * CONFIG["MAX_DURATION"]
        if len(full_audio) > max_samples:
            full_audio = full_audio[:max_samples]
        return full_audio
    return None

def transcribe(audio_data):
    """Transkribera ljud till text med optimerad prestanda från din kod"""
    if audio_data is None or isinstance(audio_data, str):
        return audio_data
    
    try:
        # VIKTIGT: Brusreducering och normalisering från din kod
        audio_tensor = torch.from_numpy(audio_data.astype(np.float32)).to(device)
        
        # Resample (behåll samma frekvens men rensa ljudet)
        audio_tensor = torchaudio.transforms.Resample(
            orig_freq=CONFIG["SAMPLE_RATE"], 
            new_freq=CONFIG["SAMPLE_RATE"]
        )(audio_tensor)
        
        # VIKTIGT: Brusreducering med lowpass filter
        audio_tensor = torchaudio.functional.lowpass_biquad(
            audio_tensor, 
            CONFIG["SAMPLE_RATE"], 
            cutoff_freq=3000
        )
        
        # VIKTIGT: Normalisering
        audio_tensor = audio_tensor / (torch.max(torch.abs(audio_tensor)) + 1e-6)
        
        # Processor med attention mask
        inputs = processor(
            audio_tensor.cpu().numpy(),
            sampling_rate=CONFIG["SAMPLE_RATE"],
            return_tensors="pt",
            return_attention_mask=True
        )
        
        input_features = inputs.input_features.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        # VIKTIGT: Använd autocast för bättre GPU-prestanda
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
        
        # Begränsa längden om för lång
        if len(transcription.split()) > CONFIG["MAX_TRANSCRIPTION_LENGTH"] // 2:
            transcription = " ".join(transcription.split()[:CONFIG["MAX_TRANSCRIPTION_LENGTH"] // 2])
        
        return transcription.strip() if transcription else None
        
    except Exception as e:
        print(f"Fel vid transkribering: {e}")
        return None

# Huvudloop
try:
    while True:
        audio = record_with_silence_detection()
        
        if isinstance(audio, str) and audio == "EXIT":
            print("\nAvslutar efter 10 sekunders tystnad...")
            break
        
        if audio is not None and not isinstance(audio, str):
            text = transcribe(audio)
            if text and text.strip():
                print(text)
                print()
                
except KeyboardInterrupt:
    print("\nAvslutar...")
finally:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()