#!/usr/bin/env python3
"""
GRACE Voice Agent - A voice-controlled virtual assistant
"""

import os
import time
import threading
import argparse
import sys
import platform
import re
from typing import Union, Optional, List, Dict, Any, Callable
import pyautogui
import keyboard
import webbrowser
# Add import for screen brightness control
import subprocess
import datetime
import traceback

# Import our custom modules
from tts import TextToSpeech
from speech_recognition_module import SpeechRecognizer
from screen_reader import ScreenReader
from system_control import SystemControl

class GraceVoiceAgent:
    def __init__(self, gemini_api_key=None, openai_api_key=None, wake_word="jarvis"):
        """Initialize the voice agent
        
        Args:
            gemini_api_key (str, optional): API key for Gemini AI
            openai_api_key (str, optional): API key for OpenAI's Whisper
            wake_word (str): Wake word to activate the agent
        """
        print("Initializing GRACE Voice Agent...")
        
        # Initialize components
        self.tts = TextToSpeech()
        self.recognizer = SpeechRecognizer(
            api="google",  # or "whisper_api" if openai_api_key
            openai_api_key=openai_api_key
        )
        self.screen_reader = ScreenReader(gemini_api_key=gemini_api_key)
        self.system = SystemControl()
        
        # Set wake word
        self.wake_word = wake_word.lower()
        
        # Store API keys
        self.gemini_api_key = gemini_api_key
        # Agent state
        self.is_running = False
        self.is_listening = False
        self.mode = "command"  # Can be "command" or "dictation"
        self.command_history = []
        
        # Available commands
        self.commands = {
            "execute": self._handle_execute_command,
            "open": self._handle_open_command,
            "type": self._handle_type_command,
            "dictate": self._handle_dictate_command,
            "stop dictation": self._handle_stop_dictation,
            "click": self._handle_click_command,
            "double click": self._handle_double_click_command,
            "right click": self._handle_right_click_command,
            "scroll": self._handle_scroll_command,
            "read": self._handle_read_command,
            "copy": self._handle_copy_command,
            "paste": self._handle_paste_command,
            "select all": self._handle_select_all_command,
            "undo": self._handle_undo_command,
            "redo": self._handle_redo_command,
            "save": self._handle_save_command,
            "close": self._handle_close_command,
            "screenshot": self._handle_screenshot_command,
            "analyze": self._handle_analyze_command,
            "debug": self._handle_debug_command,
            "change brightness": self._handle_brightness_command,
            "volume up": self._handle_volume_up_command,
            "increase volume": self._handle_volume_up_command,
            "volume down": self._handle_volume_down_command,
            "decrease volume": self._handle_volume_down_command,
            "set volume": self._handle_set_volume_command,
            "shut down": self._handle_shutdown_command,
            "shutdown": self._handle_shutdown_command,
            "shut": self._handle_shutdown_command,
            "power off": self._handle_shutdown_command,
            "turn off computer": self._handle_shutdown_command,
            "turn off": self._handle_shutdown_command,
            "exit": self._handle_exit_command,
            "quit": self._handle_exit_command,
            "help": self._handle_help_command,
        }
    
    def start(self):
        """Start the voice agent"""
        self.is_running = True
        
        print(f"Starting GRACE Voice Agent (Wake word: '{self.wake_word}')")
        self.tts.speak_async(f"GRACE Voice Agent activated. Say {self.wake_word} to control your computer. You can say {self.wake_word} execute [command] or end any command with execute to run it immediately.")
        
        try:
            # Start listening for wake word
            self._start_wake_word_detection()
            
            # Main loop
            while self.is_running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nGRACE Voice Agent interrupted by user")
        finally:
            self._cleanup()
            
    def _cleanup(self):
        """Clean up resources"""
        self.is_running = False
        self.recognizer.stop_wake_word_detection()
        print("GRACE Voice Agent stopped")
    
    def _start_wake_word_detection(self):
        """Start listening for wake word"""
        self.recognizer.start_wake_word_detection(
            wake_word=self.wake_word,
            callback=self._on_wake_word_detected
        )
    
    def _on_wake_word_detected(self):
        """Called when wake word is detected"""
        print(f"Wake word '{self.wake_word}' detected!")
        
        # Listen for the command immediately after wake word with a longer phrase time limit
        command = self.recognizer.listen_for_command(phrase_time_limit=6)
        print(f"Heard after wake word: '{command}'")
        
        if not command:
            # No command detected, provide help
            self.tts.speak_async("How can I help you?")
            self._listen_and_process_command()
            return
        
        # New handling for commands like "open chrome execute"
        if "execute" in command.lower():
            cmd_parts = command.lower().split("execute", 1)
            potential_cmd = cmd_parts[0].strip()
            print(f"Potential execute command detected: '{potential_cmd}'")
            
            if potential_cmd:
                # Process this as an execute command
                print(f"Processing as execute command: '{potential_cmd}'")
                self._handle_execute_command(potential_cmd)
                return
            
        # Check if this is a 'execute' command
        if command.lower().startswith("execute"):
            # Extract the part after 'execute'
            execute_cmd = command.lower().replace("execute", "", 1).strip()
            print(f"Detected execute command: '{execute_cmd}'")
            
            # Process the execute command immediately
            if execute_cmd:
                self._handle_execute_command(execute_cmd)
            else:
                # Just 'execute' without arguments
                self.tts.speak_async("What would you like me to execute?")
                follow_up = self.recognizer.listen_for_command(phrase_time_limit=6)
                print(f"Follow-up command: '{follow_up}'")
                if follow_up:
                    self._handle_execute_command(follow_up)
        else:
            # Regular command processing
            print(f"Regular command: '{command}'")
            self.tts.speak_async("How can I help you?")
            self._process_command(command)
    
    def _listen_and_process_command(self):
        """Listen for and process a command"""
        if self.mode == "dictation":
            text = self.recognizer.listen_for_command(phrase_time_limit=10)  # Longer time limit for dictation
            print(f"Dictation heard: '{text}'")
            if text:
                self._process_dictation(text)
        else:
            # Command mode
            command = self.recognizer.listen_for_command(phrase_time_limit=6)  # Longer time limit for commands
            print(f"Command heard: '{command}'")
            if command:
                # Check if it's a command with "execute" at the end
                if command.lower().rstrip().endswith("execute"):
                    print(f"Command with execute at the end detected: '{command}'")
                    self._process_command(command)  # The special case is handled in _process_command
                else:
                    self._process_command(command)
    
    def _process_command(self, command):
        """Process a voice command
        
        Args:
            command (str): Command to process
        """
        print(f"Processing command: '{command}'")
        self.command_history.append(command)
        
        # Check if the command ends with 'execute' to immediately execute the command
        if command.lower().rstrip().endswith("execute"):
            # Extract the command before 'execute'
            cmd_to_execute = command.lower().rstrip().rsplit("execute", 1)[0].strip()
            print(f"Command ending with execute detected: '{cmd_to_execute}'")
            
            # Process the command to execute
            if cmd_to_execute:
                # Try to match with known commands
                for cmd_keyword, handler in self.commands.items():
                    if cmd_keyword != "execute" and (cmd_keyword in cmd_to_execute or cmd_to_execute.startswith(cmd_keyword)):
                        # Extract arguments after the command keyword
                        if cmd_to_execute.startswith(cmd_keyword):
                            args = cmd_to_execute[len(cmd_keyword):].strip()
                        else:
                            parts = cmd_to_execute.split(cmd_keyword, 1)
                            args = parts[1].strip() if len(parts) > 1 else ""
                        
                        print(f"Executing command: {cmd_keyword} with args: '{args}'")
                        self.tts.speak_async(f"Executing {cmd_keyword} {args}")
                        handler(args)
                        return
                
                # If no direct match, try the general execute handler
                self._handle_execute_command(cmd_to_execute)
                return
        
        # Check for exact command matches first
        if command in self.commands:
            self.commands[command]()
            return
        
        # Check for commands that start with key phrases
        for cmd_prefix, handler in self.commands.items():
            if command.startswith(cmd_prefix):
                handler(command[len(cmd_prefix):].strip())
                return
        
        # Check for commands that contain key phrases
        for cmd_keyword, handler in self.commands.items():
            if cmd_keyword in command:
                remainder = command.replace(cmd_keyword, "", 1).strip()
                handler(remainder)
                return
        
        # No command matched
        self.tts.speak_async("I'm not sure how to help with that.")
    
    def _process_dictation(self, text):
        """Process dictated text
        
        Args:
            text (str): Dictated text
        """
        # Check if user wants to stop dictation
        if "stop dictation" in text.lower():
            self._handle_stop_dictation()
            return
        
        # Otherwise, type the text
        self.system.type_text(text)
    
    # Command handlers
    def _handle_open_command(self, args=""):
        """Handle 'open' command"""
        if not args:
            self.tts.speak_async("What would you like to open?")
            return
        
        app_name = args
        print(f"Opening {app_name}")
        
        # Speak confirmation asynchronously while executing
        self.tts.speak_async(f"Opening {app_name}")
        
        # Try to open application
        if self.system.open_application(app_name):
            # Already opened, nothing to do
            pass
        else:
            # Try as URL if application opening failed
            if "." in app_name and " " not in app_name:
                if self.system.open_url(app_name):
                    pass # Already speaking
                else:
                    self.tts.speak_async(f"Sorry, I couldn't open {app_name}")
            else:
                self.tts.speak_async(f"Sorry, I couldn't open {app_name}")
    
    def _handle_type_command(self, args=""):
        """Handle 'type' command"""
        if not args:
            self.tts.speak_async("What would you like me to type?")
            return
        
        # Perform the action while speaking
        self.tts.speak_async(f"Typing: {args[:20]}" + ("..." if len(args) > 20 else ""))
        self.system.type_text(args)
    
    def _handle_dictate_command(self, args=""):
        """Handle 'dictate' command"""
        self.tts.speak_async("Starting dictation mode. Say 'stop dictation' when you're done.")
        self.mode = "dictation"
    
    def _handle_stop_dictation(self, args=""):
        """Handle 'stop dictation' command"""
        self.mode = "command"
        self.tts.speak_async("Dictation mode stopped")
    
    def _handle_click_command(self, args=""):
        """Handle 'click' command"""
        self.tts.speak_async("Clicking")
        self.system.mouse_click()
    
    def _handle_double_click_command(self, args=""):
        """Handle 'double click' command"""
        self.tts.speak_async("Double clicking")
        self.system.double_click()
    
    def _handle_right_click_command(self, args=""):
        """Handle 'right click' command"""
        self.tts.speak_async("Right clicking")
        self.system.right_click()
    
    def _handle_scroll_command(self, args=""):
        """Handle 'scroll' command"""
        amount = 300  # Default scroll amount
        
        if "up" in args:
            self.tts.speak_async("Scrolling up")
            self.system.scroll(amount)
        elif "down" in args:
            self.tts.speak_async("Scrolling down")
            self.system.scroll(-amount)
        else:
            self.tts.speak_async("Scrolling")
            self.system.scroll(-amount)  # Default to down
    
    def _handle_read_command(self, args=""):
        """Handle 'read' command"""
        print(f"Processing read command with args: '{args}'")
        
        try:
            if "screen" in args or not args:
                self.tts.speak_async("Reading screen content. This may take a moment...")
                text = self.screen_reader.read_screen_text()
                
                if text and text.strip():
                    # Limit text to avoid very long readings
                    limited_text = text[:1000] + ("..." if len(text) > 1000 else "")
                    print(f"Read screen result ({len(text)} chars): {limited_text[:100]}...")
                    self.tts.speak_async(limited_text)
                else:
                    self.tts.speak_async("I couldn't detect any text on the screen. Make sure Tesseract OCR is installed properly.")
            elif "selection" in args or "selected" in args or "clipboard" in args:
                self.tts.speak_async("Reading selected text...")
                
                # First try using screen reader's method
                text = self.screen_reader.get_selected_text()
                
                # If that fails, try system control method
                if not text.strip():
                    text = self.system.copy_to_clipboard()
                
                if text and text.strip():
                    # Limit text to avoid very long readings
                    limited_text = text[:1000] + ("..." if len(text) > 1000 else "")
                    print(f"Read selection result ({len(text)} chars): {limited_text[:100]}...")
                    self.tts.speak_async(limited_text)
                else:
                    self.tts.speak_async("No text is selected. Please select some text first.")
            else:
                self.tts.speak_async("I'm not sure what to read. Try 'read screen' or 'read selection'.")
        except Exception as e:
            print(f"Error in read command: {e}")
            self.tts.speak_async("There was an error processing the read command.")
    
    def _handle_copy_command(self, args=""):
        """Handle 'copy' command"""
        text = self.system.copy_to_clipboard()
        if text.strip():
            self.tts.speak_async("Text copied to clipboard")
        else:
            self.tts.speak_async("No text selected to copy")
    
    def _handle_paste_command(self, args=""):
        """Handle 'paste' command"""
        self.tts.speak_async("Pasting from clipboard")
        self.system.paste_from_clipboard()
    
    def _handle_select_all_command(self, args=""):
        """Handle 'select all' command"""
        self.tts.speak_async("Selecting all")
        self.system.press_keyboard_shortcut("select all")
    
    def _handle_undo_command(self, args=""):
        """Handle 'undo' command"""
        self.tts.speak_async("Undoing")
        self.system.press_keyboard_shortcut("undo")
    
    def _handle_redo_command(self, args=""):
        """Handle 'redo' command"""
        self.tts.speak_async("Redoing")
        self.system.press_keyboard_shortcut("redo")
    
    def _handle_save_command(self, args=""):
        """Handle 'save' command"""
        self.tts.speak_async("Opening save dialog")
        self.system.press_keyboard_shortcut("save")
    
    def _handle_close_command(self, args=""):
        """Handle 'close' command"""
        if "window" in args:
            self.tts.speak_async("Closing window")
            self.system.press_keyboard_shortcut("close window")
        elif "tab" in args:
            self.tts.speak_async("Closing tab")
            self.system.press_keyboard_shortcut("close tab")
        else:
            self.tts.speak_async("Closing")
            self.system.press_keyboard_shortcut("close window")
    
    def _handle_screenshot_command(self, args=""):
        """Handle 'screenshot' command"""
        print(f"Processing screenshot command with args: '{args}'")
        
        try:
            self.tts.speak_async("Taking screenshot...")
            
            # Take screenshot using system control
            filename = self.system.take_screenshot()
            
            if filename:
                print(f"Screenshot saved to: {filename}")
                self.tts.speak_async(f"Screenshot saved to {filename}")
            else:
                # Fallback using screen reader
                screenshot = self.screen_reader.capture_screen()
                filename = self.screen_reader.save_screenshot(screenshot)
                
                if filename:
                    print(f"Screenshot saved to: {filename} (fallback method)")
                    self.tts.speak_async(f"Screenshot saved to {filename}")
                else:
                    self.tts.speak_async("Sorry, I couldn't save the screenshot.")
        except Exception as e:
            print(f"Error in screenshot command: {e}")
            self.tts.speak_async("There was an error taking the screenshot.")
    
    def _handle_analyze_command(self, args=""):
        """Handle 'analyze' command"""
        print(f"Processing analyze command with args: '{args}'")
        
        try:
            if "code" in args or "selected" in args or "selection" in args:
                self.tts.speak_async("Analyzing selected code. This may take a moment...")
                
                # First try using screen reader's method
                code = self.screen_reader.get_selected_text()
                
                # If that fails, try system control method
                if not code.strip():
                    code = self.system.copy_to_clipboard()
                
                if code and code.strip():
                    # Analyze the code
                    analysis = self.screen_reader.analyze_code(code)
                    
                    # Speak result with a brief pause
                    self.tts.speak_async("Here's my analysis:")
                    time.sleep(0.5)  # Short pause
                    
                    print(f"Code analysis result: {analysis[:100]}...")
                    self.tts.speak_async(analysis)
                else:
                    self.tts.speak_async("No code is selected for analysis. Please select some code first.")
            else:
                self.tts.speak_async("Analyzing screen content. This may take a moment...")
                
                # Capture and read screen text
                text = self.screen_reader.read_screen_text()
                
                if text and text.strip():
                    # Analyze the text as code
                    analysis = self.screen_reader.analyze_code(text)
                    
                    # Speak result with a brief pause
                    self.tts.speak_async("Here's my analysis of the screen content:")
                    time.sleep(0.5)  # Short pause
                    
                    print(f"Screen analysis result: {analysis[:100]}...")
                    self.tts.speak_async(analysis)
                else:
                    self.tts.speak_async("I couldn't detect any content to analyze on the screen.")
        except Exception as e:
            print(f"Error in analyze command: {e}")
            self.tts.speak_async("There was an error processing the analyze command.")
    
    def _handle_debug_command(self, args=""):
        """Handle 'debug' command"""
        print(f"Processing debug command with args: '{args}'")
        
        try:
            self.tts.speak_async("Debugging selected code. This may take a moment...")
            
            # First try using screen reader's method
            code = self.screen_reader.get_selected_text()
            
            # If that fails, try system control method
            if not code.strip():
                code = self.system.copy_to_clipboard()
            
            if code and code.strip():
                # Debug the code
                debug_result = self.screen_reader.debug_code(code)
                
                # Speak result with a brief pause
                self.tts.speak_async("Here's my debug analysis:")
                time.sleep(0.5)  # Short pause
                
                print(f"Debug analysis result: {debug_result[:100]}...")
                self.tts.speak_async(debug_result)
            else:
                self.tts.speak_async("No code is selected for debugging. Please select some code first.")
        except Exception as e:
            print(f"Error in debug command: {e}")
            self.tts.speak_async("There was an error processing the debug command.")
    
    def _handle_exit_command(self, args=""):
        """Handle 'exit' command"""
        self.tts.speak_async("Shutting down GRACE Voice Agent. Goodbye!")
        # Give time for exit message before shutting down
        time.sleep(1.5)
        self.is_running = False
    
    def _handle_help_command(self, args=""):
        """Handle 'help' command
        
        Args:
            args (str, optional): Command arguments
        """
        help_text = "Here are the available commands:\n"
        help_text += "- Open [app/website]: Opens applications or websites\n"
        help_text += "- Type [text]: Types text\n"
        help_text += "- Dictate: Enters dictation mode for continuous typing\n"
        help_text += "- Click, Double Click, Right Click: Performs mouse clicks\n"
        help_text += "- Scroll Up/Down: Scrolls the active window\n"
        help_text += "- Read Screen: Reads visible text on screen\n"
        help_text += "- Read Selection: Reads selected text\n"
        help_text += "- Copy, Paste, Select All: Clipboard operations\n"
        help_text += "- Undo, Redo: Undo/redo operations\n"
        help_text += "- Save: Saves the current document\n"
        help_text += "- Close: Closes the current window or tab\n"
        help_text += "- Screenshot: Takes a screenshot\n"
        help_text += "- Analyze Code: Analyzes selected code\n"
        help_text += "- Debug Code: Helps debug selected code\n"
        help_text += "- Change Brightness [1-100]: Adjusts screen brightness\n"
        help_text += "- Volume Up/Down: Increases or decreases system volume\n"
        help_text += "- Set Volume [1-100]: Sets system volume to a specific level\n"
        help_text += "- Shut Down: Shuts down your computer\n"
        help_text += "- Exit/Quit: Exits the application\n"
        help_text += "- Help: Shows this help message\n\n"
        help_text += "You can also say 'execute [command]' or end any command with 'execute' to run it immediately."
        
        print(help_text)
        self.tts.speak(help_text)

    def _handle_brightness_command(self, args=""):
        """Handle 'change brightness' command
        
        Args:
            args (str, optional): Brightness level (1-100)
        """
        # Extract the brightness level from the command
        try:
            # Extract numbers from the argument
            match = re.search(r'(\d+)', args)
            if match:
                brightness_level = int(match.group(1))
                
                # Ensure the brightness is within the valid range (1-100)
                brightness_level = max(1, min(100, brightness_level))
                
                # Set the brightness based on the platform
                if platform.system() == "Windows":
                    self._set_windows_brightness(brightness_level)
                elif platform.system() == "Darwin":  # macOS
                    self._set_macos_brightness(brightness_level)
                elif platform.system() == "Linux":
                    self._set_linux_brightness(brightness_level)
                else:
                    self.tts.speak_async("Brightness control is not supported on this platform.")
                    return
                
                self.tts.speak_async(f"Brightness set to {brightness_level} percent.")
            else:
                self.tts.speak_async("Please specify a brightness level between 1 and 100.")
        except Exception as e:
            print(f"Error setting brightness: {e}")
            self.tts.speak_async("I couldn't change the brightness. Please try again.")
    
    def _set_windows_brightness(self, level):
        """Set the brightness on Windows
        
        Args:
            level (int): Brightness level (1-100)
        """
        try:
            # PowerShell command to set brightness
            command = f'powershell "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1, {level})"'
            subprocess.run(command, shell=True)
        except Exception as e:
            print(f"Windows brightness error: {e}")
            raise
    
    def _set_macos_brightness(self, level):
        """Set the brightness on macOS
        
        Args:
            level (int): Brightness level (1-100)
        """
        try:
            # Normalized brightness value for macOS (0.0 to 1.0)
            normalized_level = level / 100.0
            command = f"osascript -e 'tell application \"System Events\" to set brightness of (get first desktop) to {normalized_level}'"
            subprocess.run(command, shell=True)
        except Exception as e:
            print(f"macOS brightness error: {e}")
            raise
    
    def _set_linux_brightness(self, level):
        """Set the brightness on Linux
        
        Args:
            level (int): Brightness level (1-100)
        """
        try:
            # Get backlight device
            backlight_dir = "/sys/class/backlight/"
            if os.path.exists(backlight_dir):
                devices = os.listdir(backlight_dir)
                if devices:
                    device_path = os.path.join(backlight_dir, devices[0])
                    
                    # Read the maximum brightness value
                    with open(os.path.join(device_path, "max_brightness"), "r") as f:
                        max_brightness = int(f.read().strip())
                    
                    # Calculate the new brightness value
                    new_brightness = int((level / 100.0) * max_brightness)
                    
                    # Set the new brightness
                    with open(os.path.join(device_path, "brightness"), "w") as f:
                        f.write(str(new_brightness))
                else:
                    raise Exception("No backlight devices found")
            else:
                raise Exception("Backlight directory not found")
        except Exception as e:
            print(f"Linux brightness error: {e}")
            raise

    def _handle_execute_command(self, args=""):
        """Handle the 'execute' command - instantly executes the rest of the command
        
        Args:
            args (str): The rest of the command to execute
        """
        if not args:
            self.tts.speak_async("Please specify what action to execute")
            return
            
        print(f"Executing command: '{args}'")
        
        # Find the first word in the remaining command
        parts = args.split(maxsplit=1)
        if not parts:
            self.tts.speak_async("I didn't understand what to execute")
            return
            
        action = parts[0].lower()
        remaining = parts[1] if len(parts) > 1 else ""
        
        # Check if this is a known command
        if action in self.commands and action != "execute":  # Prevent infinite recursion
            self.tts.speak_async(f"Executing {action} {remaining}")
            # Call the appropriate handler
            self.commands[action](remaining)
        else:
            # Try to interpret as a sentence with a command somewhere in it
            for cmd_keyword, handler in self.commands.items():
                if cmd_keyword in args and cmd_keyword != "execute":  # Prevent infinite recursion
                    # Extract the part after the command keyword
                    cmd_parts = args.split(cmd_keyword, 1)
                    if len(cmd_parts) > 1:
                        remaining = cmd_parts[1].strip()
                        self.tts.speak_async(f"Executing {cmd_keyword} {remaining}")
                        handler(remaining)
                        return
            
            # If we got here, no known command was found
            self.tts.speak_async(f"I don't know how to execute '{args}'")

    def _handle_volume_up_command(self, args=""):
        """Handle 'volume up' or 'increase volume' command
        
        Args:
            args (str, optional): Command arguments
        """
        try:
            # Extract increment amount if specified
            amount = 10  # Default increment amount
            match = re.search(r'(\d+)', args)
            if match:
                amount = min(100, int(match.group(1)))
            
            if platform.system() == "Windows":
                self._adjust_windows_volume(amount)
            elif platform.system() == "Darwin":  # macOS
                self._adjust_macos_volume(amount)
            elif platform.system() == "Linux":
                self._adjust_linux_volume(amount)
            else:
                self.tts.speak_async("Volume control is not supported on this platform.")
                return
            
            self.tts.speak_async(f"Volume increased by {amount} percent.")
        except Exception as e:
            print(f"Error adjusting volume: {e}")
            self.tts.speak_async("I couldn't adjust the volume. Please try again.")
    
    def _handle_volume_down_command(self, args=""):
        """Handle 'volume down' or 'decrease volume' command
        
        Args:
            args (str, optional): Command arguments
        """
        try:
            # Extract decrement amount if specified
            amount = 10  # Default decrement amount
            match = re.search(r'(\d+)', args)
            if match:
                amount = min(100, int(match.group(1)))
            
            if platform.system() == "Windows":
                self._adjust_windows_volume(-amount)
            elif platform.system() == "Darwin":  # macOS
                self._adjust_macos_volume(-amount)
            elif platform.system() == "Linux":
                self._adjust_linux_volume(-amount)
            else:
                self.tts.speak_async("Volume control is not supported on this platform.")
                return
            
            self.tts.speak_async(f"Volume decreased by {amount} percent.")
        except Exception as e:
            print(f"Error adjusting volume: {e}")
            self.tts.speak_async("I couldn't adjust the volume. Please try again.")
    
    def _handle_set_volume_command(self, args=""):
        """Handle 'set volume' command
        
        Args:
            args (str, optional): Volume level (1-100)
        """
        try:
            # Extract volume level from the command
            match = re.search(r'(\d+)', args)
            if match:
                volume_level = int(match.group(1))
                
                # Ensure the volume is within the valid range (0-100)
                volume_level = max(0, min(100, volume_level))
                
                # Set the volume based on the platform
                if platform.system() == "Windows":
                    self._set_windows_volume(volume_level)
                elif platform.system() == "Darwin":  # macOS
                    self._set_macos_volume(volume_level)
                elif platform.system() == "Linux":
                    self._set_linux_volume(volume_level)
                else:
                    self.tts.speak_async("Volume control is not supported on this platform.")
                    return
                
                self.tts.speak_async(f"Volume set to {volume_level} percent.")
            else:
                self.tts.speak_async("Please specify a volume level between 0 and 100.")
        except Exception as e:
            print(f"Error setting volume: {e}")
            self.tts.speak_async("I couldn't change the volume. Please try again.")
    
    def _adjust_windows_volume(self, amount):
        """Adjust the volume on Windows
        
        Args:
            amount (int): Amount to adjust volume by (-100 to 100)
        """
        try:
            # Get current volume
            get_vol_cmd = 'powershell "(Get-WmiObject -Class MSFT_MicrosoftGraphAudioPolicy).GetAudioSetting().Volume"'
            result = subprocess.run(get_vol_cmd, shell=True, capture_output=True, text=True)
            
            if result.stdout.strip():
                current_vol = float(result.stdout.strip())
                # Calculate new volume (0-100 range)
                new_vol = max(0, min(100, current_vol + amount))
                
                # Set the new volume using the keyboard module's media controls
                if amount > 0:
                    for _ in range(amount // 2):  # Each press is ~2% change
                        keyboard.send("volume up")
                else:
                    for _ in range(abs(amount) // 2):  # Each press is ~2% change
                        keyboard.send("volume down")
            else:
                # Fallback if we can't get current volume
                if amount > 0:
                    for _ in range(amount // 2):
                        keyboard.send("volume up")
                else:
                    for _ in range(abs(amount) // 2):
                        keyboard.send("volume down")
        except Exception as e:
            print(f"Windows volume adjustment error: {e}")
            # Fallback to using keyboard module directly
            try:
                if amount > 0:
                    for _ in range(amount // 2):
                        keyboard.send("volume up")
                else:
                    for _ in range(abs(amount) // 2):
                        keyboard.send("volume down")
            except:
                raise
    
    def _set_windows_volume(self, level):
        """Set the volume on Windows
        
        Args:
            level (int): Volume level (0-100)
        """
        try:
            # Try using nircmd if available
            nircmd_cmd = f'nircmd.exe setsysvolume {int(655.35 * level)}'
            try:
                subprocess.run(nircmd_cmd, shell=True)
                return
            except:
                pass  # Fall back to keyboard method
            
            # Alternate approach using keyboard simulation
            # First mute to ensure we start from 0
            keyboard.send("volume mute")
            time.sleep(0.1)
            
            # Then unmute
            keyboard.send("volume mute")
            time.sleep(0.1)
            
            # Then increase volume in steps
            for _ in range(level // 2):  # Each press is ~2% change
                keyboard.send("volume up")
                time.sleep(0.05)
        except Exception as e:
            print(f"Windows volume setting error: {e}")
            raise
    
    def _adjust_macos_volume(self, amount):
        """Adjust the volume on macOS
        
        Args:
            amount (int): Amount to adjust volume by (-100 to 100)
        """
        try:
            # Get current volume
            get_vol_cmd = "osascript -e 'output volume of (get volume settings)'"
            result = subprocess.run(get_vol_cmd, shell=True, capture_output=True, text=True)
            
            if result.stdout.strip():
                current_vol = int(result.stdout.strip())
                # Calculate new volume (0-100 range)
                new_vol = max(0, min(100, current_vol + amount))
                
                # Set the new volume
                set_vol_cmd = f"osascript -e 'set volume output volume {new_vol}'"
                subprocess.run(set_vol_cmd, shell=True)
            else:
                # Fallback if we can't get current volume
                if amount > 0:
                    for _ in range(5):  # Press 5 times
                        # Simulates pressing volume up key
                        subprocess.run("osascript -e 'tell application \"System Events\" to key code 72'", shell=True)
                else:
                    for _ in range(5):  # Press 5 times
                        # Simulates pressing volume down key
                        subprocess.run("osascript -e 'tell application \"System Events\" to key code 73'", shell=True)
        except Exception as e:
            print(f"macOS volume adjustment error: {e}")
            raise
    
    def _set_macos_volume(self, level):
        """Set the volume on macOS
        
        Args:
            level (int): Volume level (0-100)
        """
        try:
            # Set the volume directly
            command = f"osascript -e 'set volume output volume {level}'"
            subprocess.run(command, shell=True)
        except Exception as e:
            print(f"macOS volume setting error: {e}")
            raise
    
    def _adjust_linux_volume(self, amount):
        """Adjust the volume on Linux
        
        Args:
            amount (int): Amount to adjust volume by (-100 to 100)
        """
        try:
            # Check if amixer is available
            if amount > 0:
                command = f"amixer -D pulse sset Master {amount}%+"
            else:
                command = f"amixer -D pulse sset Master {abs(amount)}%-"
            
            subprocess.run(command, shell=True)
        except Exception as e:
            print(f"Linux volume adjustment error: {e}")
            
            # Fallback using pactl
            try:
                if amount > 0:
                    command = f"pactl set-sink-volume @DEFAULT_SINK@ +{amount}%"
                else:
                    command = f"pactl set-sink-volume @DEFAULT_SINK@ -{abs(amount)}%"
                
                subprocess.run(command, shell=True)
            except:
                raise
    
    def _set_linux_volume(self, level):
        """Set the volume on Linux
        
        Args:
            level (int): Volume level (0-100)
        """
        try:
            # Try with amixer first
            command = f"amixer -D pulse sset Master {level}%"
            subprocess.run(command, shell=True)
        except Exception as e:
            print(f"Linux volume setting error with amixer: {e}")
            
            # Fallback to pactl
            try:
                command = f"pactl set-sink-volume @DEFAULT_SINK@ {level}%"
                subprocess.run(command, shell=True)
            except Exception as e2:
                print(f"Linux volume setting error with pactl: {e2}")
                raise

    def _handle_shutdown_command(self, args=""):
        """Handle 'shut down' command to power off the computer
        
        Args:
            args (str, optional): Command arguments
        """
        try:
            # Clear debug output to help diagnose issues
            print("=" * 50)
            print("SHUTDOWN COMMAND RECEIVED")
            print(f"Platform: {platform.system()}")
            print(f"Args: '{args}'")
            
            # Directly proceed with shutdown without asking for confirmation
            self.tts.speak_async("Shutting down your computer. Goodbye!")
            print("Executing shutdown command...")
            
            # Small delay to allow the voice feedback to be heard
            time.sleep(3)
            
            # Execute shutdown command based on the platform
            if platform.system() == "Windows":
                # Using multiple approaches for redundancy
                print("Executing Windows shutdown command...")
                
                # Method 1: Direct system command
                os.system('shutdown /s /t 10 /f')
                
                # Method 2: Subprocess with shell=True
                # subprocess.run("shutdown /s /t 10 /f", shell=True)
                
                # Method 3: PowerShell command
                # subprocess.run(["powershell", "Stop-Computer", "-Force"], shell=True)
                
                print("Shutdown command sent")
            elif platform.system() == "Darwin":  # macOS
                print("Executing macOS shutdown command...")
                subprocess.run(["osascript", "-e", 'tell app "System Events" to shut down'], check=True)
            elif platform.system() == "Linux":
                print("Executing Linux shutdown command...")
                subprocess.run(["shutdown", "-h", "now"], check=True)
            else:
                self.tts.speak_async("Shutdown command is not supported on this platform.")
            
            print("Shutdown sequence complete")
        except Exception as e:
            print(f"Error in shutdown command: {e}")
            traceback.print_exc()  # Print full traceback
            self.tts.speak_async("I couldn't shut down your computer. Please try again or shut down manually.")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='GRACE Voice Agent')
    parser.add_argument('--gemini-key', type=str, default="AIzaSyD1VukpoEj4XVryusQckN7JnNl9y2EoQNM",
                        help='Gemini API key')
    parser.add_argument('--openai-key', type=str, default=None,
                        help='OpenAI API key (for Whisper)')
    parser.add_argument('--wake-word', type=str, default="listen",
                        help='Wake word to activate the agent')
    parser.add_argument('--voice-index', type=int, default=None,
                        help='Index of voice to use (run main.py --list-voices to see options)')
    parser.add_argument('--speech-rate', type=int, default=170,
                        help='Speech rate (words per minute)')
    parser.add_argument('--list-voices', action='store_true',
                        help='List available voices and exit')
    
    args = parser.parse_args()
    
    # If user wants to list voices
    if args.list_voices:
        tts = TextToSpeech()
        voices = tts.list_available_voices()
        print(f"Found {len(voices)} voices:")
        for voice in voices:
            print(f"Voice {voice['index']}: {voice['name']} (Gender: {voice['gender'] or 'Unknown'})")
            
            # Optionally demonstrate the voice
            tts.change_voice(voice['index'])
            tts.speak_async(f"This is the voice of {voice['name']}.")
            time.sleep(2.5)
        return
    
    agent = GraceVoiceAgent(
        gemini_api_key=args.gemini_key,
        openai_api_key=args.openai_key,
        wake_word=args.wake_word
    )
    
    # Set the voice if specified
    if args.voice_index is not None:
        agent.tts.change_voice(args.voice_index)
    
    # Set speech rate if different from default
    if args.speech_rate != 170:
        agent.tts.change_rate(args.speech_rate)
    
    agent.start()

if __name__ == "__main__":
    main()