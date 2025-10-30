import pyttsx3
import time
import threading
import sys

class TextToSpeech:
    def __init__(self, rate=170, volume=1.0, voice_index=None):
        """Initialize text-to-speech engine with customizable parameters
        
        Args:
            rate (int): Speech rate (words per minute)
            volume (float): Volume from 0.0 to 1.0
            voice_index (int, optional): Index of voice to use. None for default.
        """
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)  # Slightly slower for more natural speech
        self.engine.setProperty('volume', volume)
        
        # Use a better voice by default if available
        voices = self.engine.getProperty('voices')
        if voices:
            # Try to find a female voice for a more pleasant assistant voice
            female_voices = [i for i, v in enumerate(voices) if v.gender == 'female']
            
            # If no female voices found, look for specific voice names that tend to be better
            better_voice_index = None
            for i, voice in enumerate(voices):
                if voice.name.lower().find('zira') >= 0 or voice.name.lower().find('samantha') >= 0:
                    better_voice_index = i
                    break
            
            # Set the voice if found, otherwise use provided index or default
            if female_voices and voice_index is None:
                self.engine.setProperty('voice', voices[female_voices[0]].id)
                print(f"Using voice: {voices[female_voices[0]].name}")
            elif better_voice_index is not None and voice_index is None:
                self.engine.setProperty('voice', voices[better_voice_index].id)
                print(f"Using voice: {voices[better_voice_index].name}")
            elif voice_index is not None and 0 <= voice_index < len(voices):
                self.engine.setProperty('voice', voices[voice_index].id)
                print(f"Using voice: {voices[voice_index].name}")
            else:
                print(f"Using default voice")
        
        # For async speaking
        self.speaking_thread = None
        self.stop_speaking = False
    
    def list_available_voices(self):
        """List all available voices"""
        voices = self.engine.getProperty('voices')
        available_voices = []
        
        for i, voice in enumerate(voices):
            voice_info = {
                'index': i,
                'id': voice.id,
                'name': voice.name,
                'languages': voice.languages,
                'gender': voice.gender,
                'age': voice.age
            }
            available_voices.append(voice_info)
            
        return available_voices
    
    def speak(self, text):
        """Speak text (blocking) - Now just calls speak_async by default"""
        # For backward compatibility, but make it non-blocking by default
        return self.speak_async(text)
    
    def speak_blocking(self, text):
        """Speak text in blocking mode"""
        if not text:
            return
            
        self.engine.say(text)
        self.engine.runAndWait()
    
    def speak_async(self, text):
        """Speak text asynchronously (non-blocking)"""
        if not text:
            return
            
        if self.speaking_thread and self.speaking_thread.is_alive():
            self.stop_speaking = True
            self.speaking_thread.join(timeout=0.5)
        
        self.stop_speaking = False
        self.speaking_thread = threading.Thread(target=self._speak_thread, args=(text,))
        self.speaking_thread.daemon = True
        self.speaking_thread.start()
        
        # Return immediately
        return self.speaking_thread
    
    def _speak_thread(self, text):
        """Internal method for async speaking"""
        try:
            # Add a brief pause at the beginning and end for more natural speech
            self.engine.say(text)
            self.engine.runAndWait()
        except RuntimeError:
            # Handle error when speech is interrupted
            pass
        except Exception as e:
            print(f"Speech error: {e}", file=sys.stderr)
    
    def stop(self):
        """Stop current speech"""
        self.stop_speaking = True
        if self.speaking_thread and self.speaking_thread.is_alive():
            self.engine.stop()
    
    def change_voice(self, voice_index):
        """Change voice by index"""
        voices = self.engine.getProperty('voices')
        if 0 <= voice_index < len(voices):
            self.engine.setProperty('voice', voices[voice_index].id)
            print(f"Changed to voice: {voices[voice_index].name}")
            return True
        return False
    
    def change_rate(self, rate):
        """Change speech rate"""
        self.engine.setProperty('rate', rate)
    
    def change_volume(self, volume):
        """Change volume (0.0 to 1.0)"""
        if 0.0 <= volume <= 1.0:
            self.engine.setProperty('volume', volume)
            
    def wait_until_done(self):
        """Wait until speech is complete"""
        if self.speaking_thread and self.speaking_thread.is_alive():
            self.speaking_thread.join()

# Example usage
if __name__ == "__main__":
    tts = TextToSpeech()
    
    # List available voices
    voices = tts.list_available_voices()
    print(f"Found {len(voices)} voices:")
    for voice in voices:
        print(f"Voice {voice['index']}: {voice['name']}")
    
    # Speak text
    tts.speak("Hello, I am your voice assistant. I can help you control your computer.")
    time.sleep(2.5)  # Give time for non-blocking speech
    
    # Try different voice if available
    if len(voices) > 1:
        print("Switching to another voice...")
        tts.change_voice(1)
        tts.speak("Now I'm speaking with a different voice.")
        time.sleep(2.5)  # Give time for non-blocking speech
    
    # Demonstrate async speaking with action
    print("Demonstrating async speaking with action...")
    tts.speak_async("I'm performing actions while speaking. This makes me much more responsive.")
    
    # Do other things while speaking
    for i in range(5):
        print(f"Doing other work... {i+1}")
        time.sleep(0.5)
    
    # Wait for speaking to finish before exiting
    tts.wait_until_done()