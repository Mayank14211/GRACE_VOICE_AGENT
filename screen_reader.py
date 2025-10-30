import pytesseract
from PIL import ImageGrab, Image
import os
import tempfile
import pyautogui
import pyperclip
import time
import google.generativeai as genai
import sys
import traceback

class ScreenReader:
    def __init__(self, gemini_api_key=None):
        """Initialize screen reader
        
        Args:
            gemini_api_key (str, optional): API key for Gemini AI
        """
        self.temp_files = []
        
        # Check if Tesseract is properly installed and configured
        try:
            # Try to set the tesseract path for Windows users
            if os.name == 'nt':  # Windows
                # Common installation paths for Tesseract on Windows
                possible_paths = [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                    os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), 'Tesseract-OCR', 'tesseract.exe'),
                    os.path.join(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'), 'Tesseract-OCR', 'tesseract.exe'),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        print(f"Using Tesseract OCR from: {path}")
                        break
                else:
                    print("Warning: Tesseract OCR not found in common locations. OCR may not work.")
                    print("Install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
        except Exception as e:
            print(f"Warning: Error configuring Tesseract: {e}")
            print("Install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
        
        # Set up Gemini if API key provided
        self.gemini_model = None
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                print("Successfully connected to Gemini AI")
            except Exception as e:
                print(f"Error setting up Gemini: {e}")
                print(f"Detailed error: {traceback.format_exc()}")
    
    def __del__(self):
        """Clean up any temporary files on deletion"""
        for file in self.temp_files:
            try:
                if os.path.exists(file):
                    os.unlink(file)
            except:
                pass
    
    def capture_screen(self, region=None):
        """Capture the screen or a region of it
        
        Args:
            region (tuple, optional): Region to capture (left, top, right, bottom)
            
        Returns:
            PIL.Image: Captured image
        """
        try:
            if region:
                screenshot = ImageGrab.grab(bbox=region)
            else:
                screenshot = ImageGrab.grab()
            
            print(f"Screen captured: {screenshot.size[0]}x{screenshot.size[1]} pixels")
            return screenshot
        except Exception as e:
            print(f"Error capturing screen: {e}")
            print(f"Detailed error: {traceback.format_exc()}")
            
            # Alternative method using pyautogui as fallback
            try:
                screenshot = pyautogui.screenshot()
                print(f"Screen captured using fallback method: {screenshot.size[0]}x{screenshot.size[1]} pixels")
                return screenshot
            except Exception as fallback_error:
                print(f"Fallback screen capture also failed: {fallback_error}")
                # Return a blank image as last resort
                return Image.new('RGB', (800, 600), color='white')
    
    def save_screenshot(self, image, filename=None):
        """Save screenshot to file
        
        Args:
            image (PIL.Image): Image to save
            filename (str, optional): Filename to save to, or None for temp file
            
        Returns:
            str: Path to saved file
        """
        if not filename:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            filename = temp_file.name
            temp_file.close()
            self.temp_files.append(filename)
        
        try:
            image.save(filename)
            print(f"Screenshot saved to: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving screenshot: {e}")
            return None
    
    def extract_text_from_image(self, image):
        """Extract text from image using OCR
        
        Args:
            image (PIL.Image or str): Image or path to image
            
        Returns:
            str: Extracted text
        """
        try:
            if isinstance(image, str):
                image = Image.open(image)
            
            # Try to extract text using Tesseract OCR
            text = pytesseract.image_to_string(image)
            
            if not text.strip():
                print("OCR didn't find any text, trying with different settings...")
                # Try with different settings if no text was found
                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(image, config=custom_config)
            
            print(f"OCR extracted {len(text)} characters")
            return text
        except Exception as e:
            print(f"Error extracting text from image: {e}")
            print(f"Detailed error: {traceback.format_exc()}")
            print("Make sure Tesseract OCR is properly installed")
            return "Error: Could not extract text from screen. Make sure Tesseract OCR is installed."
    
    def read_screen_text(self, region=None):
        """Capture screen and extract text
        
        Args:
            region (tuple, optional): Region to capture
            
        Returns:
            str: Extracted text
        """
        try:
            print("Capturing screen...")
            screenshot = self.capture_screen(region)
            
            # Save screenshot temporarily for better OCR
            temp_path = self.save_screenshot(screenshot)
            
            print("Extracting text from screen...")
            text = self.extract_text_from_image(temp_path)
            
            # Clean up temp file if it exists
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                self.temp_files.remove(temp_path)
            
            return text
        except Exception as e:
            print(f"Error reading screen text: {e}")
            print(f"Detailed error: {traceback.format_exc()}")
            return "Error: Could not read screen text"
    
    def get_selected_text(self):
        """Get text currently selected on screen using clipboard
        
        Returns:
            str: Selected text
        """
        try:
            # Save current clipboard content
            current_clipboard = pyperclip.paste()
            
            # Copy selected text to clipboard
            pyautogui.hotkey('ctrl', 'c')
            time.sleep(0.5)  # Wait for clipboard to update
            
            # Get text from clipboard
            text = pyperclip.paste()
            
            # If text is the same as before, selection might have failed
            if text == current_clipboard:
                print("Warning: Selected text may not have been copied correctly")
            
            print(f"Got selected text: {len(text)} characters")
            
            # Restore original clipboard content
            pyperclip.copy(current_clipboard)
            
            return text
        except Exception as e:
            print(f"Error getting selected text: {e}")
            return ""
    
    def analyze_code(self, code):
        """Analyze code using Gemini AI
        
        Args:
            code (str): Code to analyze
            
        Returns:
            str: Analysis result
        """
        if not self.gemini_model:
            print("Warning: Gemini AI is not available. Please provide an API key.")
            return "Gemini AI is not available. Please provide an API key."
        
        if not code.strip():
            print("Warning: No code provided for analysis")
            return "No code was provided for analysis. Please select some code first."
        
        try:
            print(f"Analyzing code with Gemini AI ({len(code)} characters)...")
            prompt = f"""
            Please analyze this code and provide a brief explanation of what it does.
            Focus on the main functionality, structure, and any notable patterns or issues.
            
            CODE:
            ```
            {code}
            ```
            """
            
            response = self.gemini_model.generate_content(prompt)
            result = response.text.strip()
            print(f"Analysis complete: {len(result)} characters")
            return result
        except Exception as e:
            print(f"Error analyzing code: {e}")
            print(f"Detailed error: {traceback.format_exc()}")
            return f"Error analyzing code: {str(e)}"
    
    def debug_code(self, code):
        """Debug code using Gemini AI
        
        Args:
            code (str): Code to debug
            
        Returns:
            str: Debugging result
        """
        if not self.gemini_model:
            print("Warning: Gemini AI is not available. Please provide an API key.")
            return "Gemini AI is not available. Please provide an API key."
        
        if not code.strip():
            print("Warning: No code provided for debugging")
            return "No code was provided for debugging. Please select some code first."
        
        try:
            print(f"Debugging code with Gemini AI ({len(code)} characters)...")
            prompt = f"""
            Please debug this code, identify any errors or potential issues, and suggest fixes.
            Focus on logical errors, edge cases, performance issues, and best practices.
            
            CODE:
            ```
            {code}
            ```
            """
            
            response = self.gemini_model.generate_content(prompt)
            result = response.text.strip()
            print(f"Debug analysis complete: {len(result)} characters")
            return result
        except Exception as e:
            print(f"Error debugging code: {e}")
            print(f"Detailed error: {traceback.format_exc()}")
            return f"Error debugging code: {str(e)}"
    
    def summarize_text(self, text):
        """Summarize text using Gemini AI
        
        Args:
            text (str): Text to summarize
            
        Returns:
            str: Summarized text
        """
        if not self.gemini_model:
            print("Warning: Gemini AI is not available. Please provide an API key.")
            return "Gemini AI is not available. Please provide an API key."
        
        if not text.strip():
            print("Warning: No text provided for summarization")
            return "No text was provided for summarization."
        
        try:
            print(f"Summarizing text with Gemini AI ({len(text)} characters)...")
            prompt = f"""
            Please summarize the following text concisely, capturing the main points:
            
            {text}
            """
            
            response = self.gemini_model.generate_content(prompt)
            result = response.text.strip()
            print(f"Summarization complete: {len(result)} characters")
            return result
        except Exception as e:
            print(f"Error summarizing text: {e}")
            print(f"Detailed error: {traceback.format_exc()}")
            return f"Error summarizing text: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Initialize with Gemini API key if available
    gemini_api_key = "AIzaSyD1VukpoEj4XVryusQckN7JnNl9y2EoQNM"  # Replace with your API key
    reader = ScreenReader(gemini_api_key)
    
    # Capture screen and extract text
    print("Capturing screen...")
    text = reader.read_screen_text()
    print(f"Extracted text ({len(text)} characters):")
    print(text[:500] + "..." if len(text) > 500 else text)
    
    # Test code analysis if text is detected
    if text.strip():
        print("\nAnalyzing text as code...")
        analysis = reader.analyze_code(text)
        print("Analysis result:")
        print(analysis)