# **Auto Grader with LLM Detector**

This Python application provides educators with a comprehensive tool for both grading student assignments and detecting potential AI-generated text. It combines a natural language processing (NLP) model for text analysis with a machine learning model for LLM (Large Language Model) detection. The application features a user-friendly graphical interface (GUI) built with tkinter, making these advanced tools accessible to anyone without requiring a background in data science. It is designed to be a powerful companion for teachers, allowing them to save time and gain deeper insights into their students' work.

## **How It Works**

The core of this application lies in two primary, yet distinct, functions: automated grading and LLM text detection. These two features work in tandem to provide a holistic analysis of a student's submission. The entire process, from file upload to final report, is handled within a single, cohesive workflow.

### **1\. Automated Grading**

The grading feature works by comparing a student's submission to an answer key using a technique called **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity**. This approach provides an objective measure of content overlap, moving beyond simple keyword matching to understand the semantic similarity between two documents.

* **TF-IDF:** This method assigns a numerical value to each word in a document, indicating how important that word is to that specific document. For instance, in a history essay about the American Civil War, words like "Lincoln," "slavery," or "Gettysburg" would receive a high TF-IDF score, as they are specific to the topic. Conversely, common words like "is," "and," or "the" would receive a near-zero score, as they appear frequently in all documents and therefore have no distinguishing value. The algorithm effectively filters out noise to focus on the content that matters most.  
* **Cosine Similarity:** After all documents (the student's submission and the answer key) have been converted into numerical vectors using TF-IDF, Cosine Similarity is used to measure the angle between them in a high-dimensional space. A perfect match (identical vectors) would have an angle of 0 degrees, resulting in a score of 1.0. As the angle increases, the similarity score decreases, approaching 0\. For example, if a student's essay perfectly aligns with the content of your answer key, it will receive a grade of 100%. If it shares only a few key terms and concepts, the score will be lower. This method is particularly effective for assessing the completeness of a student's answer to a factual or conceptual question.

### **2\. LLM Detection**

The application uses a trained machine learning model to predict the likelihood that a student's work was written by an AI. The model is a **Logistic Regression classifier**, a common and effective choice for binary classification tasks like this (human vs. AI).

* **Training:** The model is trained on a curated dataset of both human-written and AI-generated text. This training process is crucial, as it allows the model to identify subtle linguistic patterns that differentiate a human author from a machine. These patterns can include things like sentence variety, use of idioms and colloquialisms, the presence of specific rhetorical structures, and the consistency of tone and style. The trained model and its vectorizer (the component that translates text into the numerical format the model can understand) are saved as llm\_detector\_model.pkl and llm\_detector\_vectorizer.pkl. This pre-trained state means the application can perform detection quickly without needing to retrain the model for every new submission.  
* **Prediction:** When a student's work is analyzed, the application feeds its text through the pre-trained vectorizer and model. The model outputs a "confidence score" between 0 and 1, which is then converted into a percentage. A score above 50% suggests that the text exhibits characteristics commonly found in AI-generated content. It is crucial to remember that this score is a diagnostic tool, not a definitive verdict. It provides an educator with a valuable data point to initiate a conversation with a student about their writing process and to encourage critical thinking and original work.

### **3\. Linguistic Analysis and Reporting**

In addition to grading and detection, the application also provides a detailed linguistic analysis of the student's submission, creating a comprehensive report for the educator.

* **Word Count:** A simple count of the words in the document, which can be useful for quickly checking if a student has met the length requirements for an assignment.  
* **Readability Score:** A calculated score based on average sentence length and lexical diversity (the variety of words used). This can give teachers insight into a student's writing complexity and clarity. For example, a low readability score could suggest that a student is using overly simple sentence structures, while a very high score might indicate dense or difficult-to-read prose. This metric can serve as a diagnostic tool for identifying areas where a student may need help with sentence construction or vocabulary.

All analysis results, including the grade, LLM confidence, and linguistic scores, are stored in a local SQLite database for future review and class-wide data visualization. This provides a secure and easily accessible record of student progress without the need for an internet connection or external servers.

## **How to Use It**

### **Prerequisites**

Before running the application, you must have Python installed. The easiest way to install all the necessary libraries is by using the requirements.txt file that is included with the script.

#### **Recommended Method: Using the requirements.txt File**

1. Navigate to the directory containing both Auto grader.py and the included requirements.txt file.  
2. Open your terminal or command prompt.  
3. Run the following command to install all the required libraries at once.  
   pip install \-r requirements.txt
   
**Note:** this program will automatically check for en_core_web_sm and then attempt to auto-install it if the libary is not found. however if you encounter an error when the program attempts to install en_core_web_sm. you will have to do it manually from your computer's command prompt or terminal you can do that by running:

python \-m spacy download en\_core\_web\_sm

### **Running the Application**

1. **Run the script:** From your terminal, navigate to the directory where you saved Auto grader.py and run the script:  
   python "Auto grader.py"

2. **Initial Setup:** The first time you run the application, it will prompt you to train the LLM detection model. This is a crucial one-time setup step. You have two options:  
   * **Upload a CSV file:** If you have a CSV file with two columns (one for text and one for labels, 0 for human and 1 for AI), you can train the model on this custom, structured data. This is ideal if you have a pre-existing dataset.

     the link to the dataset that I used can be found here: [link](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)
     
     for a more advanced dataset download it from  [here!](https://www.kaggle.com/datasets/navjotkaushal/human-vs-ai-generated-essays) 
  
   * **Select a folder:** You can select a folder containing .txt or .pdf files. The application will automatically label files based on their names (e.g., human-essay.txt or ai-report.pdf). This option is perfect for quickly training the model with your own collection of student work.  
3. **Use the GUI:** The main window provides a simple, intuitive interface for all the application's features.  
   * Enter a **Class Name** to organize your analysis results. This allows you to easily track performance for different groups of students.  
   * Click **"Upload PDF"** to select a student's submission and an answer key. The application will parse the text and prepare it for analysis.  
   * Click **"Analyze & Grade"** to generate a full report. The report will be displayed in the text box, providing the calculated grade, the LLM confidence score, and a breakdown of linguistic features like word count and readability.  
   * Use the **"Data Visualization"** section to generate graphs showing class performance or LLM confidence over time. This visual data can help you identify trends or areas of concern at a glance.  
   * Click **"Add Training Data"** to train the model with new text data. This allows you to continuously improve the model's accuracy over time.  
   * Click **"Save Report"** to export the analysis report as a .txt or .pdf file. This is useful for sharing a student's results with them or for your own record-keeping.

**note:** if you need more information about installing and using auto grader [watch this video](https://youtu.be/EeutOo8gODI)

## **Application in a Modern School Setting**

This tool can be seamlessly integrated into an educator's workflow to save time and provide valuable insights beyond a simple grade. It goes beyond a quick fix and offers a data-driven approach to teaching.

* **Efficiency:** Automates the tedious process of grading assignments, especially for subjects like English, History, or Social Studies. Imagine a history teacher with five classes, each with 25 students, who are required to write a 500-word essay. Manually grading all 125 papers can take hours. This application provides a quick and objective score based on content similarity, freeing up valuable time for educators to focus on more impactful tasks, such as lesson planning, providing individualized feedback, or designing more engaging and creative assignments.  
* **Academic Integrity:** Serves as an initial line of defense against academic dishonesty by providing a tool to flag potential instances of AI-generated work. The LLM confidence score gives teachers a data point to start a conversation with students about their work and the importance of original thought. It’s not about catching students; it's about fostering a culture of integrity and educating students on the responsible use of technology. This can lead to important discussions about critical thinking and the ethical implications of AI.  
* **Diagnostic Tool:** The linguistic analysis features can help teachers identify patterns in student writing. For example, a student with a consistently low readability score might be struggling with sentence structure or vocabulary. This tool can prompt a teacher to provide targeted support, such as recommending exercises to improve sentence complexity or introducing new vocabulary. Over time, teachers can track a student's progress and see how their writing skills are developing, providing concrete evidence for parent-teacher conferences or student support meetings.  
* **Data-Driven Instruction:** By saving analysis data by class, educators can use the visualization features to identify trends over time. For example, a teacher might notice a sudden increase in AI confidence scores in a specific class and adjust their assignment design to be more resistant to AI tools, perhaps by requiring in-class writing or personal reflection. Conversely, a teacher might notice that a particular topic consistently results in low similarity scores and realize they need to re-teach the material in a more engaging way. This allows teachers to make informed decisions and tailor their instruction to the needs of their students.

## **Disclaimer**

It is crucial to understand that this tool, particularly the LLM detection feature, is not a definitive arbiter of truth. AI detection technology is not infallible, and its predictions should be used as a guiding metric rather than a conclusive judgment. The most effective use of this tool is as a starting point for dialogue and further investigation by a human educator, whose expertise and nuanced understanding remain irreplaceable.
