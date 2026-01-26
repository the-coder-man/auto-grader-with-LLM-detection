# Auto Grader

A desktop application for educators that provides automated grading functionality for student assignments. The application offers multiple grading methods, plagiarism detection, and AI-generated text detection capabilities.

## Table of Contents

- [Features](#features)
- [Setup](#setup)
  - [Setup Script](#setup-script-recommended)
  - [Manual Installation](#manual-installation)
- [How to Use](#how-to-use)
- [Feature Details](#feature-details)
  - [PDF Upload](#pdf-upload)
  - [Keyword-Based Grading](#keyword-based-grading)
  - [AI-Assisted Grading](#ai-assisted-grading)
  - [Plagiarism Detection](#plagiarism-detection)
  - [AI Text Detection](#ai-text-detection)
  - [Report Generation](#report-generation)

---

## Features

- **Multiple Grading Methods**: Choose between keyword-based, AI-assisted grading, or plagiarism detection
- **Plagiarism Detection**: Web search and AI-powered paraphrase detection to identify copied content
- **PDF Support**: Upload and extract text from PDF documents
- **AI-Powered Grading**: Uses SmolLM2 language model for intelligent assessment
- **AI Text Detection**: Identify potentially AI-generated content in submissions
- **Detailed Reports**: Generate downloadable reports in TXT and PDF formats
- **Letter Grades**: Automatic conversion of scores to letter grades (A-F) with percentages
- **Progress Tracking**: Real-time progress indicators for long-running operations
- **Non-Blocking Interface**: Background processing keeps the application responsive

---

## Setup

### Requirements

- Python 3.11 or higher
- Internet connection (for first-time model downloads)

### Setup Script (Recommended)

The included `setup.sh` script automates the entire installation process. Run it with:

```bash
chmod +x setup.sh
./setup.sh
```

#### What the Setup Script Does

**Step 1: Operating System Detection**

The script automatically detects your operating system:
- Linux (all distributions)
- macOS (Intel and Apple Silicon)
- Windows (via Git Bash, Cygwin, or MSYS)

**Step 2: Package Manager Detection**

The script finds your system's package manager:

| Operating System | Supported Package Managers |
|------------------|---------------------------|
| Linux | apt (Debian/Ubuntu), dnf (Fedora), yum (CentOS/RHEL), pacman (Arch), zypper (openSUSE), apk (Alpine), nix |
| macOS | Homebrew, MacPorts |
| Windows | Chocolatey, Scoop, Winget |

**Step 3: Automatic Homebrew Installation (macOS)**

On macOS, if no package manager is found, the script offers to install Homebrew automatically:
- Installs Xcode Command Line Tools (required prerequisite)
- Downloads and runs the official Homebrew installer
- Configures PATH for Apple Silicon Macs (M1/M2/M3)
- Verifies the installation was successful

**Step 4: Python Installation**

If Python 3 is not found on your system:
- The script asks if you want to install it
- Uses your detected package manager to install Python
- Also installs pip and tkinter (required for the GUI)

**Step 5: Virtual Environment (Optional)**

The script asks if you want to create an isolated Python environment:
- Creates a `venv` folder in the project directory
- Activates the environment automatically
- Keeps your system Python clean
- Provides instructions for future activation

**Step 6: Dependency Installation**

Installs all required Python packages from `requirements.txt`:
- PyPDF2 (PDF text extraction)
- nltk (natural language processing)
- scikit-learn (machine learning)
- transformers & torch (AI models)
- reportlab (PDF report generation)

**Step 7: Launch Application**

After setup completes, the script offers to run the Auto Grader immediately.

#### Example Output

```
========================================
     Auto Grader Setup Script
========================================

[1/6] Detecting operating system...
      Detected: macOS

[2/6] Detecting package manager...
      Detected: Homebrew

[3/6] Checking for Python installation...
      Found: Python 3.11.5

[4/6] Virtual Environment Setup
      Would you like to create a Python virtual environment? (y/n): y
      Virtual environment created and activated!

[5/6] Installing Python dependencies...
      All dependencies installed successfully!

[6/6] Setup Complete!
========================================
      Setup completed successfully!
========================================

Would you like to run the Auto Grader application now? (y/n):
```

---

### Manual Installation

1. **Clone or download the project**

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python "auto grader.py"
   ```

### First-Time Setup Notes

- **SmolLM2 Model**: On first launch, the application will automatically download the SmolLM2 language model (~360MB). This only happens once.
- **AI Detection Model**: Before using AI text detection, you must train the model by clicking "Train AI Model". This downloads training data from HuggingFace and trains a classifier. This only needs to be done once - the model is saved for future sessions.
- **NLTK Data**: The application automatically downloads required NLTK data (punkt tokenizer) on startup.

---

## How to Use

### Basic Workflow

1. **Enter or Upload Text**: Type student text directly into the text area, or click "Upload Assignment PDF" to load a PDF document.

2. **Select Grading Method**: Choose either:
   - **Keyword-based Grading** for objective, keyword-focused assessments
   - **AI-Assisted Grading** for subjective, quality-focused evaluations

3. **Configure Grading Criteria**:
   - For keyword grading: Enter keywords and point values
   - For AI grading: Provide instructions describing your evaluation criteria

4. **Grade the Assignment**: Click "Grade Assignment" to process the submission.

5. **View Results**: Results appear in the application and a detailed report window opens automatically with download options.

---

## Feature Details

### PDF Upload

Upload PDF documents and automatically extract their text content for grading.

**How to use:**
1. Click the "Upload Assignment PDF" button
2. Select a PDF file from your computer
3. The extracted text will appear in the text input area

**Supported formats:**
- Standard PDF documents
- Multi-page PDFs (all pages are extracted)

**Note:** Scanned PDFs or image-based PDFs may not extract properly. For best results, use PDFs with selectable text.

---

### Keyword-Based Grading

A fast, objective grading method that checks for the presence of specific keywords in student submissions and awards points accordingly.

**How to use:**
1. Select "Keyword-based Grading" from the grading method options
2. Enter your keywords and point values in the format: `keyword:points`
3. Separate multiple keywords with commas
4. Click "Grade Assignment"

**Format examples:**
```
photosynthesis:10, chlorophyll:5, carbon dioxide:5
```
```
mitosis:15, cell division:10, chromosomes:5, DNA:10
```

**Scoring:**
- Each keyword found in the student text awards its full point value
- Keywords are case-insensitive
- Final score shows: Total points earned / Maximum possible points
- Percentage and letter grade are calculated automatically

**Best for:**
- Vocabulary quizzes
- Concept checks
- Technical term assessments
- Objective evaluations with clear right/wrong answers

---

### AI-Assisted Grading

An intelligent grading method that uses the SmolLM2 language model to evaluate student work based on custom criteria you provide.

**How to use:**
1. Select "AI-Assisted Grading" from the grading method options
2. Enter your grading instructions describing:
   - What criteria to evaluate
   - How to score the work
   - What constitutes good vs. poor responses
3. Click "Grade Assignment"

**Example instructions:**
```
Evaluate this essay on the American Revolution. Score out of 100 points based on:
- Historical accuracy (30 points)
- Use of specific examples (25 points)
- Clear thesis statement (20 points)
- Logical organization (15 points)
- Grammar and spelling (10 points)
```

**Output:**
- Score out of 100
- Letter grade (A, A-, B+, B, B-, C+, C, C-, D+, D, D-, F)
- Percentage
- AI-generated justification explaining the score

**Grading Scale:**
| Grade | Percentage |
|-------|------------|
| A     | 93-100%    |
| A-    | 90-92%     |
| B+    | 87-89%     |
| B     | 83-86%     |
| B-    | 80-82%     |
| C+    | 77-79%     |
| C     | 73-76%     |
| C-    | 70-72%     |
| D+    | 67-69%     |
| D     | 63-66%     |
| D-    | 60-62%     |
| F     | Below 60%  |

**Best for:**
- Essays and written responses
- Open-ended questions
- Subjective assessments
- Quality-focused evaluations

**Note:** The SmolLM2 model (~360MB) downloads automatically on first use.

---

### Plagiarism Detection

Detect potential plagiarism in student submissions by searching the web for matching content and using AI to identify paraphrased material that lacks proper citations.

**How to use:**
1. Upload or paste the student's assignment text
2. Select "Plagiarism Detection" from the grading method options
3. Enter the assignment topic (e.g., "photosynthesis in plants", "American Revolution")
4. Optionally check "Enable AI paraphrase detection" to use AI analysis (a warning will appear reminding you to review flagged content manually)
5. Click "Search the Web for Plagiarism"
6. Review the detailed report showing potential matches

**How it works:**

1. **Keyword Extraction**: The system extracts important keywords from the student text, ignoring common words like "the", "is", "and", etc.

2. **Web Search**: Uses DuckDuckGo to search for:
   - Topic + keyword combinations
   - Exact phrase matches from key sentences

3. **AI Paraphrase Detection**: If the SmolLM2 model is loaded, it analyzes sentences to detect content that appears to be paraphrased from sources without proper citation.

4. **Risk Assessment**: Generates a risk level based on findings:
   - **LOW**: No web matches and no paraphrase suspects
   - **LOW-MEDIUM**: 1-4 web matches but no paraphrase concerns
   - **MEDIUM**: 5+ web matches or 1+ paraphrase suspects
   - **HIGH**: 10+ web matches or 2+ paraphrase suspects

**Report includes:**
- Keywords analyzed
- Web search results with URLs and snippets
- Matched sentences (if exact phrases found)
- AI paraphrase analysis results
- Downloadable TXT report

**Best for:**
- Essay assignments
- Research papers
- Written reports
- Any text-based submission where originality matters

**Important disclaimer:**
Plagiarism detection is not definitive. Web search results may include:
- Legitimate sources the student properly cited
- Common knowledge or standard terminology
- False positives from similar topics

Always review flagged content manually before taking action. The AI paraphrase detection is experimental and should be used as guidance, not proof.

---

### AI Text Detection

Detect whether student submissions may contain AI-generated content using a machine learning classifier trained on millions of examples.

**Initial Setup (one-time):**
1. Click "Train AI Model"
2. Wait for the training to complete (progress bar shows status)
3. The model downloads training data from HuggingFace and trains a classifier
4. Once complete, the model is saved and ready for future sessions

**How to use:**
1. Enter or upload the text you want to analyze
2. Click "Detect AI-Generated Text"
3. View the results showing:
   - Classification (Human-written or AI-generated)
   - Confidence percentage

**Important disclaimer:**
AI detection is not foolproof. This tool provides an estimate based on patterns in the training data, but it can make mistakes. Factors like writing style, topic, and text length can affect accuracy. Use this as one tool among many when evaluating content authenticity.

**Training details:**
- Uses the `ahmadreza13/human-vs-Ai-generated-dataset` from HuggingFace
- Trains on 500,000 samples for efficiency
- Uses TF-IDF vectorization and Logistic Regression
- Model persists between sessions (no need to retrain)

---

### Report Generation

Both grading methods generate detailed reports that can be viewed and downloaded.

**Report contents:**
- Timestamp of grading
- Score and grade information
- Grading criteria used (keywords or AI instructions)
- Student submission text
- Detailed evaluation results

**For Keyword Grading:**
- List of all keywords checked
- Points earned for each keyword
- Total score and percentage
- Letter grade

**For AI Grading:**
- Letter grade with percentage
- Score out of 100
- AI-generated justification and feedback

**Download options:**
- **TXT format**: Plain text file for easy viewing and sharing
- **PDF format**: Professional formatted document suitable for records

**How to download:**
1. After grading, a report window opens automatically
2. Click "Download as TXT" or "Download as PDF"
3. Choose a save location
4. The report is saved to your selected location

---

## Technical Requirements

| Package | Minimum Version |
|---------|-----------------|
| Python | 3.11+ |
| PyPDF2 | 3.0.0 |
| nltk | 3.9.1 |
| joblib | 1.5.1 |
| scikit-learn | 1.7.1 |
| datasets | 2.0.0 |
| transformers | 4.0.0 |
| torch | 2.0.0 |
| reportlab | 4.0.0 |

---

## Files Generated

The application creates the following files during use:

- `llm_model.joblib` - Trained AI detection model (after training)
- `llm_vectorizer.joblib` - Text vectorizer for AI detection (after training)

These files allow the AI detection feature to work immediately on subsequent launches without retraining.

---

## Troubleshooting

**Problem: SmolLM2 model fails to load**
- Ensure you have a stable internet connection
- Check that you have sufficient disk space (~360MB)
- The model downloads automatically; wait for the status message to show "loaded successfully"

**Problem: PDF text extraction returns empty**
- Ensure the PDF contains selectable text (not scanned images)
- Try a different PDF viewer to verify the file isn't corrupted

**Problem: AI detection not working**
- Make sure you've trained the model first (click "Train AI Model")
- Wait for training to complete (watch the progress bar)

**Problem: Application is slow or unresponsive**
- Long operations run in background threads; the interface should remain responsive
- First-time model loading takes longer; subsequent launches are faster
