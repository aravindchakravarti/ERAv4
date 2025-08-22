# Math Snip to LaTeX

This project provides a Chrome extension that allows users to capture a portion of their screen containing an equation, perform OCR on it, and convert it into its LaTeX (MathJax) representation. The resulting LaTeX code can then be easily copied and used in LaTeX documents, Jupyter notebooks, or any other compatible environment.

## Features

*   **Screen Capture:** Easily select any part of your screen to capture an equation.
*   **Equation OCR:** Utilizes a Python backend to perform Optical Character Recognition on the captured image.
*   **LaTeX (MathJax) Conversion:** Converts the recognized equation into its LaTeX format, ready for use.
*   **Clipboard Integration:** Copy the generated LaTeX with a single click.

## Folder Structure

The project is organized into two main directories:

*   **`extension/`**: Contains the code for the Chrome extension. This includes the manifest file, popup, content scripts, and any other frontend assets.
*   **`backend/`**: Contains the Python backend code responsible for performing the OCR and LaTeX conversion.

## Getting Started

### Prerequisites

*   Google Chrome browser
*   Python (version 3.x recommended)
*   `pip` (Python package installer)

### Installation

#### 1. Backend Setup

1.  Navigate to the `backend/` directory:
    ```bash
    cd backend/
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Start the backend server (specific instructions to be added based on your backend implementation, e.g., `python app.py` or `uvicorn main:app --reload`).

#### 2. Chrome Extension Setup

1.  Open Google Chrome and navigate to `chrome://extensions`.
2.  Enable "Developer mode" by toggling the switch in the top right corner.
3.  Click on "Load unpacked" and select the `extension/` directory from your project.
4.  The "Math Snip to LaTeX" extension should now be added to your browser. You can pin it to your toolbar for easy access.

## Usage

1.  Click on the "Math Snip to LaTeX" extension icon in your Chrome toolbar.
2.  A screen capture interface will appear. Select the equation you wish to convert.
3.  The captured image will be sent to the local backend for OCR processing.
4.  Once processed, the LaTeX representation of the equation will be displayed in the extension popup.
5.  Click the "Copy" button to copy the LaTeX code to your clipboard.
6.  Paste the LaTeX code into your desired application (e.g., LaTeX editor, Jupyter Notebook, Markdown).
