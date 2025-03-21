# Email Spam Detector Chrome Extension 🛡️

A machine learning powered Chrome extension that detects spam content and provides smart summaries of long emails.

## Features ✨

- Real-time spam detection using ML model
- Smart content summarization for long emails
- Dark/Light theme support
- Export analysis reports
- View analysis history
- Scam indicator detection
- Important phrase highlighting
- Copy results to clipboard

## Tech Stack 🛠️

- Frontend: HTML, CSS, JavaScript
- Backend: Python, Flask
- ML: scikit-learn, NLTK
- Text Summarization: HuggingFace Transformers
- Chrome Extension APIs

## Setup 🚀

1. Clone the repository:
```bash
git clone https://github.com/yourusername/email-spam-detector.git
cd email-spam-detector
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file with your HuggingFace token:
   ```
   HUGGINGFACE_TOKEN=your_token_here
   ```

4. Start the Flask backend:
```bash
python model_api.py
```

5. Load the extension in Chrome:
   - Open `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select the extension directory

## Usage 📝

1. Click the extension icon on any webpage.
2. The extension will analyze the content for spam indicators.
3. View the analysis results, including:
   - Spam/Ham prediction with confidence score
   - Content summary (for longer texts)
   - Key phrase analysis
   - Scam indicators

<!-- ## Project Structure 📁

```
email-spam-detector/
├── background.js         # Background script for the extension
├── content.js            # Content script for analyzing webpages
├── icons/                # Directory containing extension icons
│   ├── icon16.png        # Icon for 16x16 resolution
│   ├── icon32.png        # Icon for 32x32 resolution
│   ├── icon48.png        # Icon for 48x48 resolution
│   └── icon128.png       # Icon for 128x128 resolution
├── model_api.py          # Flask API for spam detection and summarization
├── models/               # Directory containing ML models
│   └── spam_model.pkl    # Trained spam detection model
├── popup.html            # Popup UI
├── popup.js              # Popup logic
├── styles.css            # Styles for the popup
├── requirements.txt      # Python dependencies
├── manifest.json         # Extension manifest file
└── README.md             # Documentation
``` -->

## Contributing 🤝

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with descriptive messages
4. Submit a pull request

## License 📝

This project is licensed under the MIT License.
