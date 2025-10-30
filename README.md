# GRACE Voice Agent

GRACE (General-purpose Responsive Assistant for Computer Environment) is an AI-powered voice assistant that allows you to control your computer using voice commands. This prototype is designed to demonstrate how voice can replace traditional input methods like keyboard and mouse for human-computer interaction.

## Features

- **Speech-to-Text (STT)**: Convert your voice to text for hands-free typing
- **System Control**: Open applications, control windows, perform clicks, and more
- **Screen Reading**: Read and understand text on the screen
- **Code Analysis**: Analyze and debug code with AI assistance
- **Wake Word Detection**: Activate with customizable wake word (default: "listen")
- **Dictation Mode**: Type continuously using voice

## Requirements

- Python 3.7+
- Windows 10/11, macOS, or Linux
- Microphone
- Internet connection (for speech recognition)
- Tesseract OCR (for screen reading)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/grace-voice-agent.git
cd grace-voice-agent
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
   - **Windows**: Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt install tesseract-ocr`

4. Make sure to add Tesseract to your system PATH.

5. Verify Tesseract installation:
```bash
python tesseract_check.py
```

## Usage

Run the main application:

```bash
python main.py
```

You can customize the wake word and API keys:

```bash
python main.py --wake-word "computer" --gemini-key "your-gemini-api-key"
```

List available voices:
```bash
python main.py --list-voices
```

### Available Commands

After saying the wake word, you can use these commands:

- **System Control**:
  - "Open [application]" - Opens specified application
  - "Open [website]" - Opens a website in your default browser
  - "Type [text]" - Types the specified text
  - "Click" / "Double click" / "Right click" - Performs mouse clicks
  - "Scroll up/down" - Scrolls the current window
  - "Copy" / "Paste" / "Select all" - Performs clipboard operations
  - "Undo" / "Redo" - Performs undo/redo operations
  - "Save" - Triggers save operation
  - "Close window/tab" - Closes current window or tab
  - "Screenshot" - Takes a screenshot
  - "Change brightness [1-100]" - Adjusts the screen brightness level
  - "Volume up/down" - Increases or decreases system volume
  - "Volume up/down [amount]" - Changes volume by specified amount
  - "Set volume [0-100]" - Sets the system volume to specified level
  - "Shut down" - Immediately shuts down your computer

- **Text Input**:
  - "Dictate" - Starts dictation mode
  - "Stop dictation" - Ends dictation mode

- **Screen Reading**:
  - "Read screen" - Reads visible text on screen
  - "Read selection" - Reads text you've selected

- **Code Analysis**:
  - "Analyze code" - Analyzes selected code
  - "Debug code" - Debugs selected code

- **AI Generation**:
  - "Generate image [prompt]" - Creates an AI-generated image based on your description
  - "Generate [prompt]" - Alternative command for image generation

- **General**:
  - "Help" - Lists available commands
  - "Exit" / "Quit" - Exits the application

### Using Execute Commands

To execute commands immediately:
1. Add "execute" at the end of a command: "open notepad execute"
2. Or start with "execute": "execute open notepad"

## Screen Reading and Code Analysis Features

### Screen Reading

The screen reading feature uses Tesseract OCR to read text from your screen. For best results:

1. Make sure the text is clear and easily visible
2. Avoid complex backgrounds that might interfere with OCR
3. Use for reading error messages, dialog boxes, or other on-screen text

**Commands for Screen Reading:**
- "Read screen" - Captures the entire screen and reads the text
- "Read selection" - Reads text you've selected with the mouse

If screen reading doesn't work well, verify Tesseract is properly installed by running:
```bash
python test_screen_reader.py
```

### Code Analysis

The code analysis feature uses Google's Gemini AI to analyze and debug code:

1. First, select the code you want to analyze with your mouse
2. Then use one of the following commands:

**Commands for Code Analysis:**
- "Analyze code" - Explains what your code does, its structure, and patterns
- "Debug code" - Identifies potential errors and suggests fixes

For both commands to work:
- A valid Gemini API key must be provided
- Text must be selected before issuing the command

Example:
1. Select some code in your editor
2. Say "listen analyze code"
3. The assistant will explain what your code does

## AI Generation Features

### Image Generation

GRACE can generate images using AI models based on your text descriptions:

1. Simply say "generate image of [your description]" or "generate [your description]"
2. The AI will process your request and create an image
3. The image will be saved in the "Gen_images" directory and displayed automatically

For example:
- "Generate image of a sunset over mountains"
- "Generate a red dragon flying over a castle"

For this feature to work, you need to provide a valid Gemini API key when starting the application.

Note: Image generation requires internet access and the quality of generated images depends on the description and the AI model used.

## Troubleshooting

### Screen Reading Issues
- **No text detected**: Ensure Tesseract OCR is properly installed
- **Incorrect text**: Try selecting text manually instead of using OCR
- **Installation problems**: Run `tesseract_check.py` to diagnose issues

### Code Analysis Issues
- **"Gemini AI not available"**: Check your API key is valid and provided
- **No analysis**: Ensure you've selected code before issuing the command
- **Network errors**: Check your internet connection

## Architecture

GRACE Voice Agent is built with a modular architecture consisting of:

1. **Speech Recognition Module**: Handles wake word detection and speech-to-text
2. **Text-to-Speech Module**: Provides voice output
3. **System Control Module**: Interfaces with the operating system
4. **Screen Reader Module**: Captures and analyzes screen content
5. **Main Controller**: Orchestrates interactions between modules

## Extending GRACE

You can extend GRACE by:

1. Adding new commands in the `commands` dictionary in `GraceVoiceAgent` class
2. Creating command handlers with the naming convention `_handle_X_command`
3. Adding specialized modules for domain-specific tasks

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for Whisper API
- Google for Gemini AI API
- Tesseract OCR project
- All the open-source libraries used in this project 