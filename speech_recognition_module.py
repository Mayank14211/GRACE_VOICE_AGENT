import speech_recognition as sr
import tempfile
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import wave
import threading

class SpeechRecognizer:
    def __init__(self, api="google", openai_api_key=None, adjustment_time=0.2, timeout=3):
        """Initialize speech recognizer with specified API
        
        Args:
            api (str): Recognition API to use ("google", "whisper_api", "sphinx")
            openai_api_key (str, optional): OpenAI API key for Whisper
            adjustment_time (float): Time to adjust for ambient noise
            timeout (int): Timeout for listening
        """
        self.recognizer = sr.Recognizer()
        self.api = api
        self.openai_api_key = openai_api_key
        self.adjustment_time = adjustment_time
        self.timeout = timeout
        
        # For wake word detection
        self.wake_word_detected = False
        self.wake_word_callback = None
        self.listening_thread = None
        self.stop_listening = False
        
        # Set recognition parameters for better responsiveness
        self.recognizer.energy_threshold = 300  # Adjust based on your microphone and environment
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_adjustment_ratio = 1.5
        self.recognizer.pause_threshold = 0.5  # Shorter pause threshold for faster response
        
    def listen_once(self):
        """Listen once and return transcribed text"""
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=self.adjustment_time)
            try:
                audio = self.recognizer.listen(source, timeout=self.timeout, phrase_time_limit=5)
                return self.transcribe_audio(audio)
            except sr.WaitTimeoutError:
                print("Timeout while waiting for speech")
                return ""
            except sr.UnknownValueError:
                print("Could not understand audio")
                return ""
            except sr.RequestError as e:
                print(f"Error with the speech recognition service: {e}")
                return ""
    
    def transcribe_audio(self, audio):
        """Transcribe audio data using the selected API"""
        try:
            if self.api == "google":
                text = self.recognizer.recognize_google(audio).lower()
            elif self.api == "whisper_api" and self.openai_api_key:
                # Export audio to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                with wave.open(temp_file.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(16000)
                    wf.writeframes(audio.get_raw_data())
                
                # Use OpenAI's Whisper API
                import openai
                client = openai.OpenAI(api_key=self.openai_api_key)
                with open(temp_file.name, "rb") as file:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1", 
                        file=file
                    )
                
                # Remove temporary file
                os.unlink(temp_file.name)
                
                text = transcription.text.lower()
            elif self.api == "sphinx":
                text = self.recognizer.recognize_sphinx(audio).lower()
            else:
                print(f"Unsupported API: {self.api}, falling back to Google")
                text = self.recognizer.recognize_google(audio).lower()
            
            # Special handling for commands ending with "execute"
            # Sometimes speech recognition may not properly space "execute"
            if "execute" in text:
                # Fix cases like "openexecute" -> "open execute"
                for cmd in ["open", "type", "click", "read", "scroll", "copy", "paste", "save"]:
                    if cmd + "execute" in text:
                        text = text.replace(cmd + "execute", cmd + " execute")
            
            print(f"Recognized: {text}")
            return text
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""
    
    def record_audio(self, duration=5, sample_rate=16000):
        """Record audio for a specified duration
        
        Args:
            duration (int): Recording duration in seconds
            sample_rate (int): Sample rate in Hz
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        print(f"Recording for {duration} seconds...")
        audio_data = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1, 
                          dtype='int16')
        sd.wait()
        return audio_data, sample_rate
    
    def save_audio_to_file(self, audio_data, sample_rate, filename="temp_audio.wav"):
        """Save recorded audio to a file
        
        Args:
            audio_data (numpy.ndarray): Audio data
            sample_rate (int): Sample rate
            filename (str): Output filename
            
        Returns:
            str: Path to saved file
        """
        sf.write(filename, audio_data, sample_rate)
        return filename
        
    def start_wake_word_detection(self, wake_word="jarvis", callback=None):
        """Start listening for wake word in background
        
        Args:
            wake_word (str): Wake word to detect
            callback (function): Function to call when wake word is detected
        """
        self.wake_word = wake_word.lower()
        self.wake_word_callback = callback
        self.stop_listening = False
        
        if self.listening_thread and self.listening_thread.is_alive():
            return
            
        self.listening_thread = threading.Thread(target=self._wake_word_listener)
        self.listening_thread.daemon = True
        self.listening_thread.start()
        
    def _wake_word_listener(self):
        """Background thread for wake word detection"""
        print(f"Listening for wake word: '{self.wake_word}'")
        
        while not self.stop_listening:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.1)
                try:
                    # Shorter timeout and phrase time limit for more responsiveness
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)  # Increased the phrase time limit to catch the full command
                    try:
                        text = self.recognizer.recognize_google(audio).lower()
                        print(f"Heard: {text}")
                        
                        # Check if the wake word is in the text
                        if self.wake_word in text:
                            print(f"Wake word detected: {self.wake_word}")
                            self.wake_word_detected = True
                            
                            # Extract any text after the wake word for command processing
                            if self.wake_word_callback:
                                self.wake_word_callback()
                                
                    except sr.UnknownValueError:
                        # Speech not understood, continue listening
                        pass
                    except sr.RequestError:
                        # API error, wait and retry
                        time.sleep(0.5)  # Reduced wait time
                except:
                    # Timeout or other error, continue
                    pass
                    
            time.sleep(0.05)  # Reduced sleep time for more responsiveness
    
    def stop_wake_word_detection(self):
        """Stop wake word detection"""
        self.stop_listening = True
        if self.listening_thread:
            self.listening_thread.join(timeout=2)
            
    def change_api(self, new_api):
        """Change the speech recognition API
        
        Args:
            new_api (str): New API to use
            
        Returns:
            bool: Success status
        """
        valid_apis = ["google", "whisper_api", "sphinx"]
        if new_api in valid_apis:
            self.api = new_api
            return True
        return False

    def listen_for_command(self, prompt=None, phrase_time_limit=5):
        """Listen specifically for a longer command with a longer phrase time limit
        
        Args:
            prompt (str, optional): Optional prompt to speak before listening
            phrase_time_limit (int): Maximum duration of the phrase to capture
            
        Returns:
            str: Recognized command
        """
        with sr.Microphone() as source:
            print("Listening for command...")
            self.recognizer.adjust_for_ambient_noise(source, duration=self.adjustment_time)
            try:
                # Use a longer phrase time limit to capture the full command
                audio = self.recognizer.listen(source, timeout=self.timeout, phrase_time_limit=phrase_time_limit)
                command = self.transcribe_audio(audio)
                if command:
                    print(f"Command recognized: {command}")
                    # Check the raw text for debugging
                    print(f"Raw command text: '{command}'")
                return command
            except sr.WaitTimeoutError:
                print("Timeout while waiting for command")
                return ""
            except sr.UnknownValueError:
                print("Could not understand command")
                return ""
            except sr.RequestError as e:
                print(f"Error with the speech recognition service: {e}")
                return ""

# Example usage
if __name__ == "__main__":
    recognizer = SpeechRecognizer()
    
    # Test basic recognition
    print("Say something...")
    text = recognizer.listen_once()
    print(f"You said: {text}")
    
    # Test wake word detection
    def on_wake_word():
        print("Wake word callback triggered!")
        
    print("\nTesting wake word detection. Say 'Jarvis' to trigger.")
    recognizer.start_wake_word_detection("jarvis", on_wake_word)
    
    # Run for 30 seconds
    for i in range(30):
        print(f"Waiting... {i+1}/30")
        time.sleep(1)
        if recognizer.wake_word_detected:
            print("Wake word was detected!")
            recognizer.wake_word_detected = False
            
    recognizer.stop_wake_word_detection()