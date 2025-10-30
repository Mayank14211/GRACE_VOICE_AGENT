#!/usr/bin/env python3
"""
Tesseract OCR Installation Check

This script checks if Tesseract OCR is properly installed and working on your system.
Run this script to troubleshoot OCR functionality in the GRACE Voice Agent.
"""

import os
import sys
import subprocess
import platform
from PIL import Image, ImageDraw, ImageFont
import tempfile
import traceback

def check_tesseract():
    """Check if Tesseract OCR is properly installed and working"""
    print("Checking Tesseract OCR installation...")
    print(f"Operating System: {platform.system()} {platform.release()}")
    
    # Check if pytesseract is installed
    try:
        import pytesseract
        print("✓ pytesseract module is installed")
    except ImportError:
        print("✗ pytesseract module is not installed")
        print("  To install: pip install pytesseract")
        return False
    
    # Check Tesseract executable path
    tesseract_cmd = pytesseract.pytesseract.tesseract_cmd
    print(f"Current Tesseract command path: {tesseract_cmd}")
    
    # Look for common installation paths
    if platform.system() == 'Windows':
        common_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), 'Tesseract-OCR', 'tesseract.exe'),
            os.path.join(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'), 'Tesseract-OCR', 'tesseract.exe'),
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                print(f"Found Tesseract at: {path}")
                pytesseract.pytesseract.tesseract_cmd = path
                break
    
    # Try to run Tesseract version command
    try:
        if platform.system() == 'Windows':
            result = subprocess.run([pytesseract.pytesseract.tesseract_cmd, '--version'], 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                   text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            result = subprocess.run(['tesseract', '--version'], 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                   text=True)
            
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✓ Tesseract version: {version}")
        else:
            print(f"✗ Error running Tesseract: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Exception running Tesseract: {e}")
        if platform.system() == 'Windows':
            print("  Download Tesseract for Windows from: https://github.com/UB-Mannheim/tesseract/wiki")
        elif platform.system() == 'Darwin':  # macOS
            print("  Install Tesseract on macOS with: brew install tesseract")
        else:  # Linux
            print("  Install Tesseract on Linux with: sudo apt-get install tesseract-ocr")
        return False
    
    # Create a test image with known text
    try:
        test_text = "Hello World 123"
        print(f"Testing OCR with text: '{test_text}'")
        
        # Create a simple image with text
        img = Image.new('RGB', (200, 50), color='white')
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
            
        d.text((10, 10), test_text, fill='black', font=font)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_filename = tmp.name
            img.save(temp_filename)
            
        # OCR the image
        try:
            ocr_text = pytesseract.image_to_string(Image.open(temp_filename)).strip()
            print(f"OCR result: '{ocr_text}'")
            
            if test_text in ocr_text:
                print("✓ OCR test successful!")
            else:
                print("✗ OCR test failed - text doesn't match")
                print("  This might be due to font rendering issues")
        except Exception as ocr_error:
            print(f"✗ OCR test failed with error: {ocr_error}")
            print(traceback.format_exc())
            return False
            
        # Clean up
        try:
            os.unlink(temp_filename)
        except:
            pass
            
    except Exception as e:
        print(f"✗ Exception during OCR test: {e}")
        print(traceback.format_exc())
        return False
    
    print("\nTesseract OCR appears to be working correctly.")
    print("If you still have issues with the GRACE Voice Agent's OCR functionality:")
    print("1. Make sure the text on screen is clear and readable")
    print("2. Try selecting text manually instead of using OCR")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("  Tesseract OCR Installation Check for GRACE Voice Agent")
    print("=" * 60)
    
    check_tesseract()
    
    print("\nTo use OCR in GRACE Voice Agent:")
    print("1. Say: 'listen read screen' to read text from the screen")
    print("2. Say: 'listen read selection' to read selected text")
    
    input("\nPress Enter to exit...") 