# whisper.py

import sys
import torch
from pathlib import Path

def check_dependencies():
    """
    Verify required package dependencies.
    Returns True if all dependencies are met, False otherwise.
    """
    required_packages = ['transformers', 'torch', 'librosa']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("\nError: Missing required packages:")
        print("pip install " + " ".join(missing_packages))
        return False
    return True

def is_in_virtual_env():
    """
    Verify virtual environment status.
    """
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

if not is_in_virtual_env():
    print("\nWarning: Not running in a virtual environment!")
    print("  venv\\Scripts\\activate")
    sys.exit(1)

if not check_dependencies():
    sys.exit(1)

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import os

class AudioTranscriber:
    def __init__(self):
        """
        Initialize Whisper model and processor.
        """
        print("Loading Whisper model... This might take a few minutes on first run.")
        try:
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            self.model = self.model.to(self.device)
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Make sure you have an internet connection and enough disk space.")
            raise

    def transcribe_audio(self, audio_path):
        """
        Transcribe audio file to text.
        """
        try:
            print(f"Processing audio file: {audio_path}")
            
            print("Loading audio...")
            audio_input, sampling_rate = librosa.load(audio_path, sr=16000)
            
            print("Converting audio to model format...")
            input_features = self.processor(
                audio_input, 
                sampling_rate=sampling_rate, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            print("Generating transcription...")
            predicted_ids = self.model.generate(input_features)
            
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            print("Transcription complete!")
            return transcription
            
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return None

    def save_transcription(self, text, audio_path):
        """
        Save transcription to text file.
        """
        try:
            audio_path = Path(audio_path)
            output_path = audio_path.with_suffix('.txt')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
                
            print(f"Transcription saved to: {output_path}")
            
        except Exception as e:
            print(f"Error saving transcription: {str(e)}")

def main():
    """
    Main execution function.
    """
    project_path = Path(r"C:\Users\75\Desktop\Big Projects\Projects\Whisper")
    audio_file = "audiofile.mp3"
    audio_path = project_path / audio_file
    
    if not audio_path.exists():
        print(f"Error: Could not find {audio_path}")
        print(f"  {project_path}")
        return
    
    try:
        transcriber = AudioTranscriber()
        
        print("\nStarting transcription process...")
        transcription = transcriber.transcribe_audio(str(audio_path))
        
        if transcription:
            transcriber.save_transcription(transcription, str(audio_path))
            print("\nProcess completed successfully!")
        else:
            print("\nTranscription failed. Please check the errors above.")
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()