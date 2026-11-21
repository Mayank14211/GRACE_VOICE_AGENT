#!/usr/bin/env python3
"""
Test Script for Screen Reader and Code Analysis

This script tests the Screen Reader and Code Analysis functionality directly.
"""

import traceback
import time
import pyperclip

def test_screen_reader():
    print("=" * 60)
    print("  TESTING SCREEN READER FUNCTIONALITY")
    print("=" * 60)
    
    try:
        from screen_reader import ScreenReader
        print("✓ Imported ScreenReader module")
        
        # Gemini key comes from .env (GEMINI_API_KEY); without it, code analysis is skipped.
        from grace.config import load_settings
        key = load_settings().api_key("gemini")
        if not key:
            print("! GEMINI_API_KEY not set in .env: code analysis will report Gemini unavailable")
        reader = ScreenReader(gemini_api_key=key)
        print("✓ Initialized ScreenReader")
        
        # Test screen capture
        print("\nCapturing screen...")
        screenshot = reader.capture_screen()
        print(f"✓ Captured screenshot: {screenshot.size[0]}x{screenshot.size[1]} pixels")
        
        # Save the screenshot
        filename = reader.save_screenshot(screenshot)
        print(f"✓ Saved screenshot to: {filename}")
        
        # Test OCR
        print("\nPerforming OCR to extract text...")
        text = reader.extract_text_from_image(filename)
        if text:
            print(f"✓ Extracted {len(text)} characters of text")
            print("Sample text (first 200 chars):")
            print("-" * 50)
            print(text[:200])
            print("-" * 50)
        else:
            print("✗ No text extracted from image")
        
        # Test get_selected_text (instruct user)
        print("\nTEST: get_selected_text()")
        print("Please select some text in any window, then press Enter...")
        input()
        
        selected_text = reader.get_selected_text()
        if selected_text:
            print(f"✓ Got {len(selected_text)} characters of selected text")
            print("Sample text (first 200 chars):")
            print("-" * 50)
            print(selected_text[:200])
            print("-" * 50)
        else:
            print("✗ No selected text found or couldn't copy from clipboard")
        
        # Manual clipboard test
        backup = pyperclip.paste()
        print("\nTEST: Manual clipboard test")
        print("Placing test text in clipboard...")
        test_text = "This is a test of clipboard functionality"
        pyperclip.copy(test_text)
        time.sleep(0.5)
        clipboard_content = pyperclip.paste()
        if clipboard_content == test_text:
            print(f"✓ Clipboard test successful")
        else:
            print(f"✗ Clipboard test failed")
            print(f"Expected: '{test_text}'")
            print(f"Got: '{clipboard_content}'")
        
        # Restore original clipboard
        pyperclip.copy(backup)
        
        # Test code analysis
        print("\nTEST: analyze_code()")
        sample_code = """
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n-1)
        """
        
        print("Analyzing sample code...")
        analysis = reader.analyze_code(sample_code)
        print("Analysis result:")
        print("-" * 50)
        print(analysis)
        print("-" * 50)
        
        # Final summary
        print("\nTest Summary:")
        print("✓ Screen Reader module imported")
        print("✓ Screenshot capture tested")
        print("✓ Text extraction tested" if text else "✗ Text extraction failed")
        print("✓ Selected text copying tested" if selected_text else "✗ Selected text copying failed")
        print("✓ Clipboard functionality tested")
        print("✓ Code analysis tested" if analysis else "✗ Code analysis failed")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_screen_reader()
    print("\nPress Enter to exit...")
    input() 