import speech_recognition as sr
import pyttsx3
import pyautogui
import os
import time
import subprocess
import pyperclip
import threading
import pytesseract
from PIL import ImageGrab, Image
import google.generativeai as genai
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np
import wave
import openai

class VoiceAgent:
    def __init__(self):
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        
        # Set voice properties
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('rate', 180)  # Speed of speech
        self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        
        # You can set a specific voice if available
        # self.engine.setProperty('voice', voices[1].id)  # Index 1 is usually female voice
        
        # Initialize Gemini API
        genai.configure(api_key="AIzaSyD1VukpoEj4XVryusQckN7JnNl9y2EoQNM")
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Initialize Whisper for better speech recognition
        # using OpenAI's API (free tier has limitations)
        # Comment this out if you don't have an OpenAI API key
        # self.openai_client = openai.OpenAI(api_key="your-openai-api-key")
        
        self.is_listening = True
        self.listening_mode = "command"  # can be "command" or "dictation"
        self.dictation_text = ""
        
        # Define wake word
        self.wake_word = "jarvis"

    def speak(self, text):
        """Convert text to speech"""
        print(f"Assistant: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def listen_with_recognizer(self):
        """Listen for voice commands using SpeechRecognition"""
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=5)
                command = self.recognizer.recognize_google(audio).lower()
                print(f"User said: {command}")
                return command
            except sr.WaitTimeoutError:
                return ""
            except sr.UnknownValueError:
                return ""
            except sr.RequestError:
                self.speak("Sorry, there was an error with the speech service")
                return ""
    
    def listen_for_wake_word(self):
        """Listen specifically for the wake word"""
        with sr.Microphone() as source:
            print("Waiting for wake word...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=3)
                text = self.recognizer.recognize_google(audio).lower()
                return self.wake_word in text
            except:
                return False
    
    def listen(self):
        """Main listening function that can use different backends"""
        # For now, we're using the basic recognizer
        # In the future, you could add Whisper here for better accuracy
        return self.listen_with_recognizer()
    
    def record_audio(self, duration=5, sample_rate=16000):
        """Record audio for a specified duration"""
        print(f"Recording for {duration} seconds...")
        audio_data = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1, 
                          dtype='int16')
        sd.wait()
        return audio_data, sample_rate
    
    def save_audio_to_file(self, audio_data, sample_rate, filename="temp_audio.wav"):
        """Save recorded audio to a file"""
        sf.write(filename, audio_data, sample_rate)
        return filename
    
    def transcribe_with_whisper(self, audio_file):
        """Transcribe audio file using OpenAI's Whisper API"""
        try:
            with open(audio_file, "rb") as file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=file
                )
            return transcription.text
        except Exception as e:
            print(f"Error with Whisper API: {e}")
            return None
    
    def read_screen(self):
        """Capture and read screen content"""
        self.speak("Capturing screen content")
        screenshot = ImageGrab.grab()
        # Save screenshot to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        screenshot.save(temp_file.name)
        temp_file.close()
        
        # Use Tesseract for OCR
        text = pytesseract.image_to_string(Image.open(temp_file.name))
        
        # Clean up
        os.unlink(temp_file.name)
        
        if text.strip():
            self.speak("Here's what I found on the screen:")
            self.speak(text[:500])  # Limit the text to avoid very long responses
            return text
        else:
            self.speak("I couldn't detect any text on the screen")
            return ""

    def analyze_code_with_ai(self, code):
        """Analyze code using Gemini AI"""
        try:
            response = self.model.generate_content(
                f"Analyze this code and provide a brief explanation of what it does:\n\n{code}"
            )
            return response.text
        except Exception as e:
            print(f"Error with Gemini API: {e}")
            return "I encountered an error while analyzing the code."

    def execute_system_command(self, command):
        """Execute system commands"""
        if "open" in command:
            apps = {
                "chrome": "chrome",
                "edge": "msedge",
                "firefox": "firefox",
                "notepad": "notepad",
                "calculator": "calc",
                "explorer": "explorer",
                "word": "winword",
                "excel": "excel",
                "powerpoint": "powerpnt",
                "code": "code",
                "visual studio": "devenv",
                "terminal": "cmd",
                "command prompt": "cmd",
                "powershell": "powershell",
            }
            
            for app_name, app_command in apps.items():
                if app_name in command:
                    try:
                        subprocess.Popen(app_command)
                        self.speak(f"Opening {app_name}")
                        return
                    except Exception as e:
                        print(f"Error opening {app_name}: {e}")
                        self.speak(f"Sorry, I couldn't open {app_name}")
                        return
            
            self.speak("I don't know how to open that application")

        elif "type" in command:
            # Remove the word "type" from the command
            text_to_type = command.replace("type", "", 1).strip()
            pyautogui.write(text_to_type)
            self.speak(f"Typed: {text_to_type}")

        elif "dictate" in command:
            self.speak("Starting dictation mode. Say 'stop dictation' when you're done.")
            self.listening_mode = "dictation"
            self.dictation_text = ""
            return
        
        elif "scroll" in command:
            if "up" in command:
                pyautogui.scroll(300)
                self.speak("Scrolling up")
            elif "down" in command:
                pyautogui.scroll(-300)
                self.speak("Scrolling down")

        elif "click" in command:
            pyautogui.click()
            self.speak("Clicked")
            
        elif "double click" in command:
            pyautogui.doubleClick()
            self.speak("Double clicked")
            
        elif "right click" in command:
            pyautogui.rightClick()
            self.speak("Right clicked")

        elif "read screen" in command or "read the screen" in command:
            self.read_screen()
            
        elif "screenshot" in command:
            screenshot = ImageGrab.grab()
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            screenshot.save(f"screenshot_{timestamp}.png")
            self.speak("Screenshot saved")
            
        elif "copy" in command:
            pyautogui.hotkey('ctrl', 'c')
            self.speak("Copied to clipboard")
            
        elif "paste" in command:
            pyautogui.hotkey('ctrl', 'v')
            self.speak("Pasted from clipboard")
            
        elif "save" in command:
            pyautogui.hotkey('ctrl', 's')
            self.speak("Save dialog opened")
            
        elif "undo" in command:
            pyautogui.hotkey('ctrl', 'z')
            self.speak("Undone")
            
        elif "redo" in command:
            pyautogui.hotkey('ctrl', 'y')
            self.speak("Redone")
            
        elif "select all" in command:
            pyautogui.hotkey('ctrl', 'a')
            self.speak("Selected all")

    def process_coding_commands(self, command):
        """Handle coding-related commands"""
        if "debug" in command:
            self.speak("Starting debug analysis...")
            
            # Get selected code or text on screen
            pyautogui.hotkey('ctrl', 'c')
            time.sleep(0.5)  # Give time for the copy operation to complete
            code = pyperclip.paste()
            
            if code.strip():
                analysis = self.analyze_code_with_ai(code)
                self.speak("Here's my analysis:")
                self.speak(analysis)
            else:
                self.speak("No code was selected for debugging")

        elif "explain code" in command:
            self.speak("Analyzing the selected code...")
            
            # Get selected code
            pyautogui.hotkey('ctrl', 'c')
            time.sleep(0.5)  # Give time for the copy operation to complete
            code = pyperclip.paste()
            
            if code.strip():
                explanation = self.analyze_code_with_ai(code)
                self.speak("Here's my explanation:")
                self.speak(explanation)
            else:
                self.speak("No code was selected for explanation")

    def process_dictation(self, text):
        """Process text in dictation mode"""
        if "stop dictation" in text:
            self.listening_mode = "command"
            self.speak("Dictation ended")
            return
        
        # Add the text to the accumulating dictation
        if self.dictation_text:
            self.dictation_text += " " + text
        else:
            self.dictation_text = text
            
        # Type the dictated text
        pyautogui.write(text + " ")

    def run(self):
        """Main loop for the voice agent"""
        self.speak("Voice agent activated. Say 'Jarvis' to activate me.")
        
        while self.is_listening:
            # Wait for wake word in continuous mode
            if self.listen_for_wake_word():
                # Wake word detected
                self.speak("How can I help you?")
                
                # Listen for command
                command = self.listen()
                
                if not command:
                    continue
                
                # Exit command
                if "exit" in command or "stop" in command or "quit" in command:
                    self.speak("Shutting down voice agent")
                    self.is_listening = False
                    break
                
                # Process the command based on current mode
                if self.listening_mode == "dictation":
                    self.process_dictation(command)
                else:
                    # Process regular commands
                    if any(word in command for word in ["open", "type", "scroll", "click", "read", "screenshot", 
                                                      "copy", "paste", "save", "undo", "redo", "select", "dictate"]):
                        self.execute_system_command(command)
                    elif any(word in command for word in ["debug", "explain"]):
                        self.process_coding_commands(command)
                    else:
                        self.speak("I'm not sure how to handle that command")

def main():
    print("Starting Voice Agent...")
    agent = VoiceAgent()
    try:
        agent.run()
    except KeyboardInterrupt:
        print("Voice Agent stopped by user")

if __name__ == "__main__":
    main()