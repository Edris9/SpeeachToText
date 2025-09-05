import torch
import torchaudio
import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import logging
import threading
from typing import Optional, Union
import sys
import queue

# Konfigurationssektion
CONFIG = {
    "MODEL_NAME": "openai/whisper-small",  # Byt till small för bättre noggrannhet
    "SAMPLE_RATE": 16000,
    "SILENCE_THRESHOLD": 0.008,  # Ökat för att undvika klippning
    "MAX_DURATION": 12,  # Ökat till 12 sekunder för att fånga hela meningen
    "SILENCE_AFTER_SPEECH": 1.2,  # Längre tystnad för att fånga hela tal
    "INITIAL_SILENCE_TIMEOUT": 5,
    "EXIT_SILENCE_TIMEOUT": 10,
    "CHUNK_DURATION": 0.2,  # Snabb respons
    "LANGUAGE": "swedish",
    "MAX_TRANSCRIPTION_LENGTH": 50,  # Begränsa output
}

# Konfigurera loggning
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self):
        """Initialisera Whisper-modellen och processorn."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Använder enhet: {self.device}")
        if self.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        try:
            self.processor = WhisperProcessor.from_pretrained(CONFIG["MODEL_NAME"])
            self.model = WhisperForConditionalGeneration.from_pretrained(CONFIG["MODEL_NAME"])
            self.model.config.pad_token_id = self.model.config.eos_token_id
            self.model.config.forced_decoder_ids = None
            self.model.to(self.device)
            self.model.eval()
            if torch.__version__ >= "2.0" and self.device == "cuda":
                self.model = torch.compile(self.model)
        except Exception as e:
            logger.error(f"Fel vid laddning av modell: {e}")
            raise
        
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()

    def record_with_silence_detection(self) -> Optional[Union[np.ndarray, str]]:
        """Spela in ljud med tystnadsdetektion."""
        chunk_size = int(CONFIG["SAMPLE_RATE"] * CONFIG["CHUNK_DURATION"])
        audio_chunks = []
        speech_detected = False
        silence_after_speech = 0
        total_silence = 0
        no_input_counter = 0

        logger.info("AI lyssnar...")
        
        stream = sd.InputStream(
            samplerate=CONFIG["SAMPLE_RATE"],
            channels=1,
            dtype="float32",
            blocksize=chunk_size
        )
        
        try:
            with stream:
                for _ in range(int(CONFIG["MAX_DURATION"] / CONFIG["CHUNK_DURATION"])):
                    if self.stop_event.is_set():
                        return None
                    chunk, _ = stream.read(chunk_size)
                    chunk = np.squeeze(chunk)
                    
                    volume = np.max(np.abs(chunk))
                    logger.debug(f"Chunk volym: {volume:.4f}")
                    
                    if volume > CONFIG["SILENCE_THRESHOLD"]:
                        speech_detected = True
                        silence_after_speech = 0
                        audio_chunks.append(chunk)
                        no_input_counter = 0
                        total_silence = 0
                    else:
                        if speech_detected:
                            audio_chunks.append(chunk)
                            silence_after_speech += CONFIG["CHUNK_DURATION"]
                            if silence_after_speech > CONFIG["SILENCE_AFTER_SPEECH"]:
                                break
                        else:
                            total_silence += CONFIG["CHUNK_DURATION"]
                            if total_silence >= CONFIG["INITIAL_SILENCE_TIMEOUT"] and no_input_counter == 0:
                                logger.info("Inget input...")
                                no_input_counter = 1
                                return None
                            if total_silence >= CONFIG["EXIT_SILENCE_TIMEOUT"]:
                                return "EXIT"
        
        except Exception as e:
            logger.error(f"Fel vid inspelning: {e}")
            return None
        
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            max_samples = CONFIG["SAMPLE_RATE"] * CONFIG["MAX_DURATION"]
            if len(full_audio) > max_samples:
                full_audio = full_audio[:max_samples]
            logger.debug(f"Ljudlängd: {len(full_audio)/CONFIG['SAMPLE_RATE']:.2f} sekunder")
            return full_audio
        return None

    def transcribe(self, audio_data: Optional[np.ndarray]) -> Optional[str]:
        """Transkribera ljud till text med optimerad prestanda."""
        if audio_data is None or isinstance(audio_data, str):
            return audio_data
        
        try:
            # Brusreducering och normalisering
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32)).to(self.device)
            audio_tensor = torchaudio.transforms.Resample(
                orig_freq=CONFIG["SAMPLE_RATE"], new_freq=CONFIG["SAMPLE_RATE"]
            )(audio_tensor)
            # Brusreducering
            audio_tensor = torchaudio.functional.lowpass_biquad(
                audio_tensor, CONFIG["SAMPLE_RATE"], cutoff_freq=3000
            )
            # Normalisera
            audio_tensor = audio_tensor / (torch.max(torch.abs(audio_tensor)) + 1e-6)
            
            inputs = self.processor(
                audio_tensor.cpu().numpy(),
                sampling_rate=CONFIG["SAMPLE_RATE"],
                return_tensors="pt",
                return_attention_mask=True
            )
            input_features = inputs.input_features.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            with torch.no_grad(), torch.amp.autocast(device_type=self.device):
                predicted_ids = self.model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    language=CONFIG["LANGUAGE"],
                    task="transcribe",
                    max_new_tokens=CONFIG["MAX_TRANSCRIPTION_LENGTH"],
                    num_beams=1,
                )
            
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            if len(transcription.split()) > CONFIG["MAX_TRANSCRIPTION_LENGTH"] // 2:
                transcription = " ".join(transcription.split()[:CONFIG["MAX_TRANSCRIPTION_LENGTH"] // 2])
            return transcription.strip() if transcription else None
        
        except Exception as e:
            logger.error(f"Fel vid transkribering: {e}")
            return None

    def process_audio_queue(self):
        """Bearbeta ljud från kön i en separat tråd."""
        while not self.stop_event.is_set():
            try:
                audio = self.audio_queue.get(timeout=0.5)
                if isinstance(audio, str) and audio == "EXIT":
                    break
                if audio is not None:
                    text = self.transcribe(audio)
                    if text:
                        logger.info("Transkription: %s", text)
                self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Fel i köbearbetning: {e}")

    def run(self):
        """Kör huvudloopen för inspelning och transkription."""
        logger.info("Systemet startat. Väntar på tal...")
        logger.info("(Avslutas automatiskt efter %d sekunders tystnad)", CONFIG["EXIT_SILENCE_TIMEOUT"])
        
        transcription_thread = threading.Thread(target=self.process_audio_queue, daemon=True)
        transcription_thread.start()
        
        try:
            while not self.stop_event.is_set():
                audio = self.record_with_silence_detection()
                if isinstance(audio, str) and audio == "EXIT":
                    self.audio_queue.put("EXIT")
                    break
                if audio is not None:
                    self.audio_queue.put(audio)
        
        except KeyboardInterrupt:
            logger.info("Avslutar via användaravbrott...")
        finally:
            self.stop_event.set()
            transcription_thread.join()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def main():
    """Huvudfunktion för att starta transkriberaren."""
    try:
        transcriber = WhisperTranscriber()
        transcriber.run()
    except Exception as e:
        logger.error(f"Kritiskt fel: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()