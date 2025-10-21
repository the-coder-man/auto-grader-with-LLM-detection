**Auto Grader** 

**Overview and Core Purpose** 

The Auto Grader is a highly practical desktop application built entirely with **Python** and the native **Tkinter** library. It is designed to be a powerful, cross-platform utility for anyone managing and assessing large volumes of written content—specifically educators across K-12 and higher education, corporate trainers running certification programs, and content quality managers in publishing. 

Its core mission is to solve the pervasive problem of subjective and time-consuming manual grading, which often leads to instructor burnout and delays in student feedback. The application achieves this by offering a flexible, **dual-pronged assessment system** that allows the user to choose the right tool for the job: 

1\. A rapid, objective **Keyword-Based Grading** engine, perfect for factual recall, technical document review, and checklist-style assessments. This mode offers immediate, black-and-white scoring based purely on the presence of essential terms. 

2\. A more nuanced, sophisticated **AI-Assisted Grading** method, which uses specialized, modern AI technology to evaluate complex criteria like critical thinking, structural coherence, persuasive argument strength, and overall writing quality. This goes beyond simple counting to offer true analytical feedback. 

Crucially, the Auto Grader directly addresses modern challenges in content integrity by integrating a robust **AI-Generated Text Detection** module. This feature is a vital tool for maintaining academic and content integrity by helping users objectively identify submissions potentially created by Large Language Models (LLMs). By automating the preliminary review, scoring, and originality checks, the Auto Grader aims to dramatically streamline the assessment workflow. This frees up instructors to focus their energy on creating more engaging lessons and providing meaningful, personalized support to students, fundamentally transforming grading from a repetitive burden into a focused, objective process. The resulting benefit is not just a substantial time savings, but also ensuring **consistency and fairness** in scoring across entire classes and teams. 

**Setup and Installation**

**Note: if you need more infomation about how to install Auto grader please refer to this: [YouTube video](https://m.youtube.com/watch?v=dqzF_r_3B9w&t=10s)**

To get the Auto Grader up and running smoothly, you need a stable **Python environment** (version 3.8 or higher is highly recommended). Since the application relies on Python's standard libraries and highly portable, cross-platform machine learning tools, it is fully compatible with **Windows, macOS, and Linux** operating systems. 

**1\. Save Files and Virtual Environment Setup (Recommended)** Before installing any project dependencies, it is best practice to create and activate a **virtual**  
**environment**. This is essential professional practice because it isolates the project's required libraries, preventing version conflicts with other Python software you may have installed on your system. 

● **Step 1.1:** Ensure the main script (auto grader.py) and the dependency list (requirements.txt) are saved together in a dedicated project folder (e.g., AutoGrader\_Project). 

● **Step 1.2:** Open your terminal or command prompt in that specific folder and run the following commands to set up and enter the isolated virtual environment: \# Create the environment (name it 'venv') 

python \-m venv venv 

\# Activate the environment 

\# For Windows: 

.\\venv\\Scripts\\activate 

\# For macOS/Linux: 

source venv/bin/activate 

Once successfully activated, your terminal prompt will change to show (venv) at the beginning, confirming that any subsequent installations will be confined to this isolated project environment. 

**2\. Install Dependencies** 

All necessary open-source libraries—which range from the GUI framework to the machine learning core and PDF generation tools—are meticulously listed in the requirements.txt file. You can install everything required with one simple command. 

pip install \-r requirements.txt 

This single command will pull the required versions of scikit-learn, PyPDF2, torch, transformers, and the rest of the stack. 

*Note on NLTK Data and Efficiency:* The application relies on the Natural Language Toolkit (**NLTK**) for essential text processing tasks. Specifically, the application automatically attempts to download the punkt dataset during its very first execution. This component is crucial for **tokenization**—the process of accurately breaking down large bodies of text into smaller, meaningful units like sentences and words. Accurate tokenization is foundational for both the precise keyword matching and the sophisticated numerical analysis required by the AI models. 

**3\. Run the Application**  
With your virtual environment active and all dependencies verified and installed, you can launch the desktop application from the terminal: 

python "auto grader.py" 

The application's native graphical user interface (GUI) window will open immediately. This provides a clean workspace ready for assignment input and the selection of your desired grading method. If the application encounters an error during startup, the most common issues relate to dependency installation or the virtual environment not being active; in such a 

case, please double-check those two points. 

**Core Features and How to Use** 

The application is engineered around a structured, four-step workflow designed for user efficiency: **Input, Selection, Processing, and Reporting.** 

**Assignment Input and Text Management** 

The interface prioritizes maximum flexibility for getting the student's text into the system quickly: 

● **Direct Text Entry:** This is the quickest option for immediate analysis. Use the large, scrollable text input area for grading very short responses, checking code comments, or analyzing text copied from an email or a web source. The ScrolledText widget handles large documents gracefully. 

● **PDF Upload (Automation Focus):** The "Upload Assignment PDF" button handles the most common academic submission format. The application uses the robust **PyPDF2** library to automatically read and extract the clean, raw text from the document. This feature is a massive time-saver, as it intelligently bypasses common formatting issues found in PDFs, eliminating the need for the user to manually copy/paste, which can sometimes introduce errors. 

**1\. Keyword-Based Grading (Quantitative and Objective)** 

This is the system's objective, quantitative assessment mode, ensuring absolute consistency in scoring based purely on the presence or absence of specific content. 

● **How it Works (The Preprocessing Pipeline):** After selecting the **Keyword-based** radio button, the system initiates a rigorous preprocessing step. This typically includes converting all submitted text to lowercase and removing common punctuation (normalization). This ensures a robust, **case-insensitive matching** process. For example, the system will correctly recognize and score "Chlorophyll," "CHLOROPHYLL," and "chlorophyll" equally. 

● **Input and Scoring Logic (The Grading Recipe):** The user defines the exact terms and their assigned values (e.g., photosynthesis:10, chlorophyll:5, light-dependent:15). This  
input acts as a digital checklist with a dynamic scoring system: if the keyword is detected anywhere in the submission, the corresponding points are instantly awarded. This method is perfect for large classes where grading consistency is paramount. 

● **Result and Feedback:** A concise, easily digestible report is generated. Critically, it does more than just show the total score; it explicitly lists *which* of the required keywords were detected and, just as importantly, *which were missing*. This provides clear, unambiguous, binary feedback on factual recall, making it easy for students to understand exactly why points were deducted. 

**2\. AI-Assisted Grading (Qualitative and Nuanced)** 

This is the most advanced, qualitative method designed for assessing nuanced writing skills, critical thinking, and adherence to complex, multi-layered rubrics. 

● **The AI Engine:** The system utilizes the **SmolLM2-360M-Instruct** model. The strategic choice of a *small* language model is a key architectural decision: this model is optimized for **rapid inference** and **local operation**, meaning the sophisticated analysis is both fast and guarantees that the grading process does not rely on external, third-party cloud services for maximum data privacy and low latency. 

● **Input and Custom Criteria (Rubric Execution):** You must provide detailed, **multi-faceted instructions** that accurately reflect a full, human-designed grading rubric. These instructions can cover abstract concepts. Examples include: "Evaluate the tone for professionalism (0-10 points)," "Assess the logical flow and transition quality between paragraphs (0-40 points)," or a comprehensive breakdown like: "Grade on clarity of thesis (40 points), evidence supporting the argument (30 points), adherence to Chicago style citation format (20 points), and strength of the conclusion (10 points). The maximum score is 100 points." 

● **Result and High-Quality Feedback:** The AI model processes the text against these weighted, detailed instructions. It delivers a precise numeric score, but the most powerful element is the **targeted textual feedback**. This narrative explains *why* the student received their score, often pointing out specific structural weaknesses (e.g., "Weak transition in paragraph 3") or areas where evidence was strong. This detailed, immediate, and high-quality feedback significantly enhances a student's revision and learning process. 

**3\. AI-Generated Text Detection (Integrity and Verification)** 

This feature is a critical security layer aimed at helping users uphold academic honesty and content originality by providing an objective, statistical assessment of text origin. 

● **Prerequisite (The Crucial One-Time Training):** The heart of this feature is a custom-trained machine learning classifier. You **must** click the "Train LLM Detection Model" button once. This process is complex and time-intensive due to the necessary data download and training, but it runs in a **non-blocking thread**, ensuring the GUI remains responsive while the work is done in the background. The training pipeline involves:  
1\. **Data Acquisition:** Downloading the extensive 

**ahmadreza13/human-vs-Ai-generated-dataset** (a large-scale, categorized dataset). 

2\. **Vectorization:** Using the **TfidfVectorizer** to convert millions of text examples into numerical feature vectors. This mathematical process captures the statistical importance and frequency of words, essential for pattern recognition. 

3\. Model Training: Training a high-performance classifier (like Logistic Regression or SGD) to find subtle, distinguishing patterns between human writing styles and the synthetic outputs of LLMs. 

Once training is complete, the model files (llm\_model.joblib and llm\_vectorizer.joblib) are saved locally using joblib and automatically loaded on every subsequent launch, making the detection process instant thereafter. 

● **Detection Process and Interpretation:** Click **Detect AI Text** to analyze the text currently in the input box. The system provides a clear prediction ("AI-Generated" or "Human-Written") and a decisive **confidence score** (e.g., 92% confidence). Users are strongly advised to interpret this as guidance: a high confidence score simply flags a text for a closer **human investigation** and discussion with the student, as no AI detection tool is 100% accurate or infallible. This promotes transparency and fair practice. 

**4\. Detailed Report Generation (Archiving and Communication)** 

Record-keeping and clear communication of results are absolutely essential for any formal assessment. A detailed, multi-section report is generated automatically after every successful grading cycle. 

● **Report Content:** The report is comprehensive, capturing the assignment's metadata, the chosen grading method (Keyword or AI), the final score, and a complete breakdown of the results. This includes either the explicit list of found/missing keywords or the full narrative feedback and analysis provided by the AI model. 

● **Export Options:** The report can be instantly archived in two industry-standard formats for maximum utility: 

○ **TXT File:** A clean, minimal file containing the raw text of the report. This is perfect for quick data export, importing into spreadsheets, or viewing in any basic text editor. ○ **PDF File:** A professional, formatted document generated using the **ReportLab** library. This output is ideal for official school records, emailing to students or parents, or printing out for paper records, ensuring a polished presentation. 

**Potential Real-World Applications** 

The powerful, dual functionality of the Auto Grader means it is useful far beyond a traditional classroom environment: 

● Higher Education and K-12 (Enhanced): 

For objective tasks (e.g., medical terminology quizzes, history dates, science definitions), the Keyword Grader offers instant assessment. For complex tasks like college-level  
research papers, the AI-Assisted mode can be used to check against a comprehensive list of formal requirements (e.g., presence of an abstract, specific section headers, or a minimum complexity score), significantly reducing the instructor's initial review time. This allows the teacher to focus their limited time on the quality of ideas, not the basic structure. 

● Corporate Training and HR (Expanded): 

In professional settings, the Keyword Grader is perfect for screening high volumes of job applications or legal contracts for the mandatory presence of required technical skills, certifications, or regulatory clauses. The AI-Assisted feature can grade the clarity, persuasiveness, and effectiveness of employee reports, summaries, or internal communication documents based on company-specific tone and style guidelines, ensuring consistent internal quality. 

● Content and Publishing Quality Assurance (Critical Screening): 

In fast-paced marketing or journalism, the AI Detection module is invaluable. It can be used to pre-screen articles, web copy, and commissioned content for originality before publication. Ensuring that all published material is genuinely human-generated helps to maintain brand reputation, ethical publishing standards, and is increasingly crucial for effective search engine optimization (SEO) performance. 

● Self-Study and Tutoring Centers (Empowerment): 

This application can empower students and learners directly. By allowing them to use the AI-Assisted feature to check their own drafts against the exact criteria of their teacher's rubric, they receive instant, private, and non-judgmental feedback. This helps them improve their revision skills, internalize complex rubric requirements, and develop better critical thinking before the final submission deadline, effectively turning the grading tool into a highly personalized, 24/7 writing tutor. 

**Credits and External Components** 

The application's advanced capabilities are a powerful example of what can be built using the open-source community, relying on a robust, well-maintained stack of Python libraries and specialized models. 

**Core Libraries (The Desktop Interface and I/O)** 

● **tkinter:** The foundational Python standard library for all graphical interface elements. It ensures a stable, responsive user experience without needing external, complex GUI frameworks. 

● **PyPDF2:** The essential utility for handling PDF files, enabling the smooth and reliable extraction of raw text from student submissions. 

● **reportlab:** This library is dedicated to high-quality document creation, specifically programmatically generating the professional, finalized PDF reports from the grading results. 

● **joblib:** Crucial for **model persistence**. It allows the application to efficiently save the large trained detection model and the TfidfVectorizer to disk, and then instantly load  
them back up on subsequent runs, avoiding the lengthy one-time training on every user launch. 

**Machine Learning Stack (The Data and Logic)** 

● **scikit-learn:** The backbone of the detection and keyword systems. It provides all the classic machine learning tools, including the high-speed **Logistic Regression** classifier and the powerful **TfidfVectorizer** for turning text into numerical feature vectors by weighting words based on their frequency and importance across the dataset. 

● **nltk (Natural Language Toolkit):** Provides the necessary foundational natural language processing tools, particularly the punkt tokenizer, ensuring accurate word and sentence splitting across diverse texts for reliable analysis. 

● **datasets, transformers, and torch:** These libraries form the necessary infrastructure for running the advanced, instruction-tuned AI model. **torch (PyTorch)** provides the deep learning framework that efficiently executes the complex tensor calculations of the smaller **SmolLM2** model, ensuring fast performance even on standard desktop hardware. 

**External Models and Training Data** 

● **AI-Assisted Grading Model:** The core AI engine is the **SmolLM2-360M-Instruct** model, which is instruction-tuned to accurately follow specific, nuanced scoring rules and custom rubrics provided by the user. 

● **AI Detection Training Data:** The binary classifier for detecting AI-generated text is trained on the extensive **ahmadreza13/human-vs-Ai-generated-dataset**. This large, high-quality dataset, sourced from the Hugging Face ecosystem, is what makes the detection model robust and accurate by providing it with millions of examples of both human and machine-generated writing styles.
