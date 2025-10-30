import subprocess
import os
import pyautogui
import time
import pyperclip
import webbrowser

class SystemControl:
    def __init__(self):
        """Initialize system control module"""
        # Default applications dictionary
        self.applications = {
            "chrome": "chrome",
            "edge": "msedge",
            "firefox": "firefox",
            "notepad": "notepad",
            "calculator": "calc",
            "explorer": "explorer",
            "file explorer": "explorer",
            "word": "winword",
            "excel": "excel",
            "powerpoint": "powerpnt",
            "code": "code",
            "visual studio code": "code",
            "visual studio": "devenv",
            "terminal": "cmd",
            "command prompt": "cmd",
            "powershell": "powershell",
            # Add more applications as needed
        }
        
        # Store actions history
        self.action_history = []
        
    def open_application(self, app_name):
        """Open an application by name
        
        Args:
            app_name (str): Name of the application to open
            
        Returns:
            bool: Success status
        """
        app_name = app_name.lower()
        
        # Special handling for common websites
        websites = {
            "youtube": "https://youtube.com",
            "google": "https://google.com",
            "facebook": "https://facebook.com",
            "twitter": "https://twitter.com",
            "instagram": "https://instagram.com",
            "linkedin": "https://linkedin.com",
            "github": "https://github.com",
            "gmail": "https://mail.google.com",
            "amazon": "https://amazon.com",
            "netflix": "https://netflix.com",
            "whatsapp": "https://web.whatsapp.com",
            "spotify": "https://open.spotify.com",
        }
        
        # Check if it's a website
        if app_name in websites:
            return self.open_url(websites[app_name])
        
        # Check if app name is in our dictionary
        if app_name in self.applications:
            app_command = self.applications[app_name]
            try:
                subprocess.Popen(app_command, shell=True)
                self.action_history.append(f"Opened {app_name}")
                print(f"Successfully launched {app_name}")
                return True
            except Exception as e:
                print(f"Error opening {app_name}: {e}")
                
                # Try alternate methods for common applications
                try:
                    os.system(f"start {app_command}")
                    self.action_history.append(f"Opened {app_name} (alternative method)")
                    print(f"Successfully launched {app_name} using alternative method")
                    return True
                except Exception as alt_error:
                    print(f"Alternative method also failed: {alt_error}")
                    return False
        
        # If app name contains a dot, treat it as a URL
        if '.' in app_name:
            return self.open_url(app_name)
        
        # If we don't have a predefined command, try to open it directly
        try:
            os.system(f"start {app_name}")
            self.action_history.append(f"Opened {app_name}")
            print(f"Opened {app_name} using generic method")
            return True
        except Exception as e:
            print(f"Generic open method failed: {e}")
            return False
    
    def open_url(self, url):
        """Open a URL in the default browser
        
        Args:
            url (str): URL to open
            
        Returns:
            bool: Success status
        """
        # Add http:// if not present
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
            
        try:
            webbrowser.open(url)
            self.action_history.append(f"Opened URL: {url}")
            return True
        except Exception as e:
            print(f"Error opening URL: {e}")
            return False
    
    def type_text(self, text):
        """Type text using keyboard
        
        Args:
            text (str): Text to type
        """
        try:
            # Try to use pyautogui's write function
            pyautogui.write(text)
            print(f"Typed text: {text[:20]}..." if len(text) > 20 else f"Typed text: {text}")
            
            # Alternative approach using clipboard for reliability with special characters
            if not text.isascii():
                # Backup current clipboard
                old_clipboard = pyperclip.paste()
                
                # Set new content and paste
                pyperclip.copy(text)
                pyautogui.hotkey('ctrl', 'v')
                
                # Restore clipboard after a small delay
                time.sleep(0.5)
                pyperclip.copy(old_clipboard)
                print("Used clipboard method for non-ASCII text")
                
            self.action_history.append(f"Typed: {text[:20]}..." if len(text) > 20 else f"Typed: {text}")
        except Exception as e:
            print(f"Error typing text: {e}")
            
            # Fallback method
            try:
                pyperclip.copy(text)
                pyautogui.hotkey('ctrl', 'v')
                print("Used fallback clipboard method for typing")
            except Exception as fallback_error:
                print(f"Fallback typing method also failed: {fallback_error}")
    
    def press_key(self, key):
        """Press a keyboard key
        
        Args:
            key (str): Key to press
        """
        pyautogui.press(key)
        self.action_history.append(f"Pressed key: {key}")
    
    def press_key_combination(self, *keys):
        """Press a combination of keys
        
        Args:
            *keys: Keys to press simultaneously
        """
        pyautogui.hotkey(*keys)
        self.action_history.append(f"Pressed keys: {'+'.join(keys)}")
    
    def press_keyboard_shortcut(self, shortcut):
        """Press a common keyboard shortcut by name
        
        Args:
            shortcut (str): Name of the shortcut
            
        Returns:
            bool: Success status
        """
        shortcuts = {
            "copy": ["ctrl", "c"],
            "paste": ["ctrl", "v"],
            "cut": ["ctrl", "x"],
            "undo": ["ctrl", "z"],
            "redo": ["ctrl", "y"],
            "save": ["ctrl", "s"],
            "select all": ["ctrl", "a"],
            "find": ["ctrl", "f"],
            "new": ["ctrl", "n"],
            "print": ["ctrl", "p"],
            "refresh": ["f5"],
            "close tab": ["ctrl", "w"],
            "close window": ["alt", "f4"],
            "switch window": ["alt", "tab"],
            "window left": ["win", "left"],
            "window right": ["win", "right"],
            "maximize": ["win", "up"],
            "minimize": ["win", "down"],
            "start menu": ["win"],
            "task view": ["win", "tab"],
            "screenshot": ["win", "shift", "s"],
        }
        
        shortcut_lower = shortcut.lower()
        
        # Check for exact matches
        if shortcut_lower in shortcuts:
            try:
                keys = shortcuts[shortcut_lower]
                self.press_key_combination(*keys)
                print(f"Executed keyboard shortcut: {shortcut} ({'+'.join(keys)})")
                return True
            except Exception as e:
                print(f"Error executing shortcut {shortcut}: {e}")
                
                # Try individual key presses as fallback
                try:
                    for key in shortcuts[shortcut_lower]:
                        pyautogui.keyDown(key)
                    time.sleep(0.2)
                    for key in reversed(shortcuts[shortcut_lower]):
                        pyautogui.keyUp(key)
                    print(f"Used fallback method for shortcut: {shortcut}")
                    return True
                except Exception as fallback_error:
                    print(f"Fallback shortcut method also failed: {fallback_error}")
                    return False
        
        # Check for partial matches
        for name, keys in shortcuts.items():
            if name in shortcut_lower or shortcut_lower in name:
                try:
                    self.press_key_combination(*keys)
                    print(f"Executed similar keyboard shortcut: {name} ({'+'.join(keys)})")
                    return True
                except Exception as e:
                    print(f"Error executing similar shortcut {name}: {e}")
        
        print(f"Unknown shortcut: {shortcut}")
        return False
    
    def mouse_click(self, x=None, y=None, button="left"):
        """Click the mouse
        
        Args:
            x (int, optional): X coordinate, or None for current position
            y (int, optional): Y coordinate, or None for current position
            button (str): Button to click ("left", "right", "middle")
        """
        try:
            if x is not None and y is not None:
                pyautogui.click(x, y, button=button)
                self.action_history.append(f"{button.capitalize()} clicked at ({x}, {y})")
            else:
                pyautogui.click(button=button)
                current_pos = pyautogui.position()
                print(f"{button.capitalize()} click at position {current_pos}")
                self.action_history.append(f"{button.capitalize()} clicked at current position {current_pos}")
        except Exception as e:
            print(f"Error clicking mouse: {e}")
            
            # Fallback for click
            try:
                if button == "left":
                    pyautogui.mouseDown(button=button)
                    time.sleep(0.1)
                    pyautogui.mouseUp(button=button)
                    print("Used fallback method for mouse click")
                elif button == "right":
                    self.press_key_combination('shift', 'f10')
                    print("Used keyboard shortcut fallback for right click")
            except Exception as fallback_error:
                print(f"Fallback click method also failed: {fallback_error}")
    
    def double_click(self, x=None, y=None):
        """Double-click the mouse
        
        Args:
            x (int, optional): X coordinate, or None for current position
            y (int, optional): Y coordinate, or None for current position
        """
        if x is not None and y is not None:
            pyautogui.doubleClick(x, y)
            self.action_history.append(f"Double-clicked at ({x}, {y})")
        else:
            pyautogui.doubleClick()
            current_pos = pyautogui.position()
            self.action_history.append(f"Double-clicked at current position {current_pos}")
    
    def right_click(self, x=None, y=None):
        """Right-click the mouse
        
        Args:
            x (int, optional): X coordinate, or None for current position
            y (int, optional): Y coordinate, or None for current position
        """
        self.mouse_click(x, y, button="right")
    
    def scroll(self, amount):
        """Scroll the mouse wheel
        
        Args:
            amount (int): Amount to scroll (positive for up, negative for down)
        """
        try:
            pyautogui.scroll(amount)
            direction = "up" if amount > 0 else "down"
            print(f"Scrolled {direction} (amount: {abs(amount)})")
            self.action_history.append(f"Scrolled {direction}")
        except Exception as e:
            print(f"Error scrolling: {e}")
            
            # Alternative method using keyboard
            try:
                if amount > 0:
                    for _ in range(abs(amount) // 100):
                        pyautogui.press('pageup')
                else:
                    for _ in range(abs(amount) // 100):
                        pyautogui.press('pagedown')
                print("Used keyboard fallback for scrolling")
            except Exception as fallback_error:
                print(f"Fallback scroll method also failed: {fallback_error}")
    
    def move_mouse(self, x, y):
        """Move the mouse to a position
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
        """
        pyautogui.moveTo(x, y)
        self.action_history.append(f"Moved mouse to ({x}, {y})")
    
    def get_mouse_position(self):
        """Get current mouse position
        
        Returns:
            tuple: (x, y) coordinates
        """
        return pyautogui.position()
    
    def copy_to_clipboard(self):
        """Copy selected text to clipboard
        
        Returns:
            str: Copied text
        """
        pyautogui.hotkey('ctrl', 'c')
        time.sleep(0.5)  # Wait for clipboard to update
        text = pyperclip.paste()
        self.action_history.append("Copied text to clipboard")
        return text
    
    def paste_from_clipboard(self):
        """Paste from clipboard"""
        pyautogui.hotkey('ctrl', 'v')
        self.action_history.append("Pasted from clipboard")
    
    def set_clipboard(self, text):
        """Set clipboard content
        
        Args:
            text (str): Text to set
        """
        pyperclip.copy(text)
        self.action_history.append("Set clipboard content")
    
    def take_screenshot(self, filename=None):
        """Take a screenshot
        
        Args:
            filename (str, optional): Filename to save to, or None for timestamp
            
        Returns:
            str: Path to saved screenshot
        """
        if not filename:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        self.action_history.append(f"Took screenshot: {filename}")
        return filename
    
    def get_action_history(self, limit=10):
        """Get recent action history
        
        Args:
            limit (int): Maximum number of actions to return
            
        Returns:
            list: Recent actions
        """
        return self.action_history[-limit:]

# Example usage
if __name__ == "__main__":
    control = SystemControl()
    
    # Test opening an application
    print("Opening Notepad...")
    control.open_application("notepad")
    time.sleep(2)
    
    # Type some text
    print("Typing text...")
    control.type_text("This is a test of the system control module")
    
    # Take a screenshot
    print("Taking screenshot...")
    screenshot = control.take_screenshot()
    print(f"Screenshot saved to {screenshot}")
    
    # Close notepad
    print("Closing Notepad...")
    control.press_key_combination("alt", "f4")
    time.sleep(1)
    control.press_key("n")  # Don't save
    
    # Show action history
    print("\nAction history:")
    for action in control.get_action_history():
        print(f"- {action}")