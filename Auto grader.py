import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import spacy
import datetime
import os
import pickle
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import warnings
import subprocess
import sys

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Global Data and Model ---
db_name = 'llm_detector.db'
model_path = 'llm_detector_model.pkl'
vectorizer_path = 'llm_detector_vectorizer.pkl'
model = LogisticRegression()
vectorizer = TfidfVectorizer()

# --- Database Functions ---
def setup_database():
    """Sets up the SQLite database and tables if they don't exist."""
    try:
        with sqlite3.connect(db_name) as conn:
            cursor = conn.cursor()
            # Table for storing documents for future training
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY,
                    text TEXT NOT NULL,
                    label INTEGER NOT NULL  -- 0 for human, 1 for AI
                )
            ''')
            # Table for storing analysis results (updated to include grade_score)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY,
                    class_name TEXT NOT NULL,
                    grade_score REAL,
                    llm_confidence REAL,
                    readability REAL,
                    word_count INTEGER,
                    analysis_date TEXT
                )
            ''')
            conn.commit()
    except sqlite3.Error as e:
        messagebox.showerror("Database Error", f"Failed to set up database: {e}")

def save_training_data_to_db(text, label):
    """Saves a single document and its label to the training data table."""
    try:
        with sqlite3.connect(db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO training_data (text, label) VALUES (?, ?)', (text, label,))
            conn.commit()
    except sqlite3.Error as e:
        messagebox.showerror("Database Error", f"Failed to save training data: {e}")

def save_analysis_to_db(class_name, grade_score, llm_confidence, readability, word_count):
    """Saves the results of a single analysis to the database."""
    try:
        with sqlite3.connect(db_name) as conn:
            cursor = conn.cursor()
            analysis_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute('''
                INSERT INTO analyses (class_name, grade_score, llm_confidence, readability, word_count, analysis_date)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (class_name, grade_score, llm_confidence, readability, word_count, analysis_date))
            conn.commit()
    except sqlite3.Error as e:
        messagebox.showerror("Database Error", f"Failed to save analysis data: {e}")

def get_training_data():
    """Retrieves all training data from the database."""
    try:
        with sqlite3.connect(db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT text, label FROM training_data')
            data = cursor.fetchall()
            return data
    except sqlite3.Error as e:
        messagebox.showerror("Database Error", f"Failed to retrieve training data: {e}")
        return []

def get_class_data(class_name):
    """Retrieves all analysis data for a specific class."""
    try:
        with sqlite3.connect(db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT grade_score, llm_confidence, readability, analysis_date FROM analyses WHERE class_name = ?', (class_name,))
            data = cursor.fetchall()
            return data
    except sqlite3.Error as e:
        messagebox.showerror("Database Error", f"Failed to retrieve class data: {e}")
        return []

# --- Text Preprocessing and Feature Extraction ---

def get_text(file_path):
    """Extracts text from a given file path (supports .txt and .pdf) using PyPDF2."""
    if not os.path.exists(file_path):
        messagebox.showerror("File Error", f"File not found: {file_path}")
        return None

    try:
        if file_path.lower().endswith('.pdf'):
            text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text()
            return text
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            messagebox.showerror("File Error", "Unsupported file format. Please upload a .txt or .pdf file.")
            return None
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read file: {e}")
        return None
def preprocess_text_for_grading(text, nlp):
    """Preprocesses text for vectorization using spaCy."""
    if not text:
        return ""
    try:
        doc = nlp(text)
        cleaned_words = [
            token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha
        ]
        return " ".join(cleaned_words)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return ""

def calculate_linguistic_features(doc):
    """Calculates linguistic features using spaCy."""
    word_count = len([token for token in doc if token.is_alpha])

    # Calculate average sentence length and lexical diversity
    sentences = list(doc.sents)
    avg_sentence_length = word_count / len(sentences) if len(sentences) > 0 else 0

    unique_words = set([token.text.lower() for token in doc if token.is_alpha])
    lexical_diversity = len(unique_words) / word_count if word_count > 0 else 0

    # A simple readability score
    readability = (avg_sentence_length + lexical_diversity * 10) / 2

    return readability, word_count

# --- Machine Learning Model Training ---
def train_model():
    """Trains the ML model on all data from the database."""
    all_data = get_training_data()
    if not all_data or len(all_data) < 2:
        return "Not enough data to train model. Please upload a training dataset."

    texts = [row[0] for row in all_data]
    labels = [row[1] for row in all_data]

    global vectorizer
    global model

    try:
        # Vectorize the text data
        vectorizer.fit(texts)
        X = vectorizer.transform(texts)
        y = np.array(labels)

        # Train the model
        model.fit(X, y)

        # Save the trained model and vectorizer
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)

        return "Model retrained successfully."
    except Exception as e:
        return f"Model retraining failed: {e}"

def load_model():
    """Loads a pre-trained ML model and vectorizer."""
    global model
    global vectorizer
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            return True
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load the trained model: {e}")
            return False
    return False

# --- TKinter App Class ---
class AutoGraderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Auto Grader with LLM Detector")
        self.geometry("1000x800")
        self.configure(bg="#f4f7f9")

        self.nlp = None
        self.is_dependencies_loaded = False
        self.student_file_path = None
        self.answer_key_file_path = None

        self.create_widgets()
        setup_database()
        self.load_dependencies()

    def load_dependencies(self):
        """Loads spaCy model, installing it if necessary, using a conditional subprocess command."""
        self.status_label.config(text="Status: Loading spaCy model...", fg="orange")
        self.update_idletasks()
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.is_dependencies_loaded = True
            self.status_label.config(text="Status: Ready", fg="green")
            self.check_initial_setup()
        except OSError:
            messagebox.showinfo("spaCy Model", "Downloading spaCy 'en_core_web_sm' model. This may take a moment.")
            try:
                # Use subprocess for a more robust installation
                if sys.version_info.major == 3:
                    python_executable = "python3"
                else:
                    python_executable = "python"
                
                subprocess.run([python_executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                
                # After successful download, load the model
                self.nlp = spacy.load("en_core_web_sm")
                self.is_dependencies_loaded = True
                self.status_label.config(text="Status: Ready", fg="green")
                self.check_initial_setup()
            except subprocess.CalledProcessError as e:
                messagebox.showerror("spaCy Error", f"Failed to download spaCy model: {e}")
                self.status_label.config(text="Status: Failed to load dependencies.", fg="red")
                self.nlp = None
                self.is_dependencies_loaded = False
            except Exception as e:
                messagebox.showerror("spaCy Error", f"An unexpected error occurred: {e}")
                self.status_label.config(text="Status: Failed to load dependencies.", fg="red")
                self.nlp = None
                self.is_dependencies_loaded = False

    def check_initial_setup(self):
        """Checks if a model exists and prompts the user to train one if not."""
        if not load_model():
            self.prompt_for_training_data()

    def prompt_for_training_data(self):
        """Asks the user to upload a training dataset and trains the model."""
        choice = messagebox.askquestion("Initial Setup", "Welcome! It seems this is your first time running the app. We need to train the LLM detection model. Would you like to upload a CSV file? If not, you can select a folder with text files.")

        if choice == 'yes':
            file_path = filedialog.askopenfilename(title="Select Training Data CSV File", filetypes=[("CSV Files", "*.csv")])
            if file_path:
                self.process_csv_training_data(file_path)
            else:
                messagebox.showwarning("Training Canceled", "Model training canceled. The application will not be able to detect LLM text without a trained model.")
        else:
            folder_path = filedialog.askdirectory(title="Select Training Data Folder")
            if folder_path:
                self.process_folder_training_data(folder_path)
            else:
                messagebox.showwarning("Training Canceled", "Model training canceled. The application will not be able to detect LLM text without a trained model.")

    def process_csv_training_data(self, file_path):
        """Loads training data from a CSV file and saves it to the DB."""
        try:
            df = pd.read_csv(file_path)

            text_col = simpledialog.askstring("CSV Column", "Enter the name of the column containing the text:")
            label_col = simpledialog.askstring("CSV Column", "Enter the name of the column containing the labels (0 for human, 1 for AI):")

            if text_col and label_col and text_col in df.columns and label_col in df.columns:
                for index, row in df.iterrows():
                    text_content = str(row[text_col])
                    label = int(row[label_col])
                    save_training_data_to_db(text_content, label)

                train_status = train_model()
                messagebox.showinfo("Training Complete", train_status)
            else:
                messagebox.showerror("CSV Format Error", "One or both of the specified column names were not found in the CSV file.")
        except Exception as e:
            messagebox.showerror("CSV Read Error", f"Failed to read CSV file: {e}")

    def process_folder_training_data(self, folder_path):
        """Iterates through files in the selected folder and saves them to the DB."""
        if not os.path.isdir(folder_path):
            messagebox.showerror("Invalid Folder", "The selected path is not a valid folder.")
            return

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.txt', '.pdf')):
                label = -1
                if "human" in filename.lower():
                    label = 0  # Human-written
                elif "ai" in filename.lower() or "llm" in filename.lower():
                    label = 1  # AI-generated

                if label != -1:
                    try:
                        text_content = get_text(file_path)
                        if text_content:
                            save_training_data_to_db(text_content, label)
                    except Exception as e:
                        print(f"Failed to process file {filename}: {e}")

        train_status = train_model()
        messagebox.showinfo("Training Complete", train_status)

    def create_widgets(self):
        """Builds the GUI elements for the application."""
        # Main frames
        main_frame = tk.Frame(self, bg="#f4f7f9", padx=15, pady=15)
        main_frame.pack(fill="both", expand=True)

        input_frame = tk.Frame(main_frame, bg="#ffffff", padx=15, pady=15, bd=2, relief="groove")
        input_frame.pack(fill="x", pady=10)

        output_frame = tk.Frame(main_frame, bg="#ffffff", padx=15, pady=15, bd=2, relief="groove")
        output_frame.pack(fill="both", expand=True, pady=10)

        graph_controls_frame = tk.Frame(main_frame, bg="#ffffff", padx=15, pady=15, bd=2, relief="groove")
        graph_controls_frame.pack(fill="x", pady=10)

        self.graph_display_frame = tk.Frame(main_frame, bg="#ffffff")
        self.graph_display_frame.pack(fill="both", expand=True, pady=(0,10))

        # Status Label
        self.status_label = tk.Label(input_frame, text="Status: Loading...", fg="gray", bg="#ffffff")
        self.status_label.grid(row=0, column=2, padx=5, pady=5, sticky="e")

        # Input widgets
        tk.Label(input_frame, text="Class Name:", bg="#ffffff").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.class_name_entry = tk.Entry(input_frame, width=30)
        self.class_name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        tk.Label(input_frame, text="Student Test:", bg="#ffffff").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.upload_student_button = tk.Button(input_frame, text="Upload PDF", command=lambda: self.upload_pdf("student"))
        self.upload_student_button.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.student_file_label = tk.Label(input_frame, text="No file selected", bg="#ffffff")
        self.student_file_label.grid(row=1, column=2, padx=5, pady=5, sticky="w")

        tk.Label(input_frame, text="Answer Key:", bg="#ffffff").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.upload_answer_key_button = tk.Button(input_frame, text="Upload PDF", command=lambda: self.upload_pdf("answer_key"))
        self.upload_answer_key_button.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.answer_key_file_label = tk.Label(input_frame, text="No file selected", bg="#ffffff")
        self.answer_key_file_label.grid(row=2, column=2, padx=5, pady=5, sticky="w")

        self.analyze_button = tk.Button(input_frame, text="Analyze & Grade", command=self.analyze_and_grade)
        self.analyze_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.retrain_button = tk.Button(input_frame, text="Add Training Data", command=self.prompt_for_training_data)
        self.retrain_button.grid(row=3, column=2, pady=10)

        # Output widgets
        self.report_label = tk.Label(output_frame, text="Analysis Report:", bg="#ffffff", font=("Helvetica", 12, "bold"))
        self.report_label.pack(pady=(0, 10))
        self.report_text = tk.Text(output_frame, wrap="word", bg="#f4f7f9")
        self.report_text.pack(fill="both", expand=True)
        self.report_text.tag_config("green_tag", foreground="green")
        self.report_text.tag_config("red_tag", foreground="red")

        # Graphing widgets
        tk.Label(graph_controls_frame, text="Data Visualization:", bg="#ffffff", font=("Helvetica", 12, "bold")).pack(side="left", padx=5)

        tk.Label(graph_controls_frame, text="Select Graph Type:", bg="#ffffff").pack(side="left", padx=5)
        self.graph_type = tk.StringVar(self)
        self.graph_type.set("Grade Scores")
        self.graph_menu = tk.OptionMenu(graph_controls_frame, self.graph_type, "Grade Scores", "LLM Confidence", "Grade vs Confidence", "Readability vs Confidence")
        self.graph_menu.pack(side="left", padx=5)

        self.generate_graph_button = tk.Button(graph_controls_frame, text="Generate Graph", command=self.generate_graph)
        self.generate_graph_button.pack(side="left", padx=5)

        # Save Report button
        self.save_button = tk.Button(main_frame, text="Save Report", command=self.save_report_to_file)
        self.save_button.pack(pady=10)

    def upload_pdf(self, file_type):
        """Opens a file dialog to select a PDF file."""
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            if file_type == "student":
                self.student_file_path = file_path
                self.student_file_label.config(text=os.path.basename(file_path))
            elif file_type == "answer_key":
                self.answer_key_file_path = file_path
                self.answer_key_file_label.config(text=os.path.basename(file_path))

    def analyze_and_grade(self):
        """Performs test grading and LLM text detection and updates the GUI."""
        if not self.is_dependencies_loaded:
            messagebox.showwarning("Dependencies Not Loaded", "Please wait for the necessary models to load before analyzing.")
            return
        
        # Additional defensive check to ensure nlp model is loaded
        if self.nlp is None:
            messagebox.showerror("Model Error", "spaCy model failed to load properly. Please restart the application.")
            return

        if not self.student_file_path or not self.answer_key_file_path:
            messagebox.showwarning("Missing Files", "Please upload both a student test and an answer key.")
            return

        if not load_model():
            messagebox.showwarning("No Model", "No trained model found. Please train a model by uploading a training dataset.")
            return

        # 1. Grade the test using TF-IDF and Cosine Similarity
        student_text = get_text(self.student_file_path)
        answer_key_text = get_text(self.answer_key_file_path)

        if not student_text or not answer_key_text:
            return

        documents = [preprocess_text_for_grading(student_text, self.nlp), preprocess_text_for_grading(answer_key_text, self.nlp)]
        if not documents[0] or not documents[1]:
            messagebox.showwarning("Empty Text", "Could not extract sufficient text from one or both of the documents.")
            return

        grader_vectorizer = TfidfVectorizer()
        tfidf_matrix = grader_vectorizer.fit_transform(documents)
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        grade_score = cosine_sim[0][0] * 100 # Convert to percentage

        # 2. Perform LLM detection using the trained model
        student_doc = self.nlp(student_text)
        readability_score, word_count = calculate_linguistic_features(student_doc)

        confidence = 0
        try:
            text_vector = vectorizer.transform([student_text])
            llm_confidence = model.predict_proba(text_vector)[0][1]
            confidence = llm_confidence * 100
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed to predict LLM confidence: {e}")
            return

        # 3. Store results in DB
        class_name = self.class_name_entry.get() or "Default Class"
        save_analysis_to_db(class_name, grade_score, confidence, readability_score, word_count)

        # 4. Generate Report
        self.report_text.delete("1.0", tk.END)
        self.report_text.insert(tk.END, f"--- Analysis Report for '{os.path.basename(self.student_file_path)}' ---\n\n")

        self.report_text.insert(tk.END, "--- Test Grading ---\n")
        self.report_text.insert(tk.END, f"Grade Score: {grade_score:.2f}%\n")

        self.report_text.insert(tk.END, "\n--- LLM Detection Analysis ---\n")
        self.report_text.insert(tk.END, f"LLM Confidence Score: {confidence:.2f}%\n")
        if confidence > 50:
            self.report_text.insert(tk.END, "Prediction: Likely AI-generated\n", "red_tag")
        else:
            self.report_text.insert(tk.END, "Prediction: Likely Human-written\n", "green_tag")

        self.report_text.insert(tk.END, "\n--- Linguistic Analysis ---\n")
        self.report_text.insert(tk.END, f"Readability Score: {readability_score:.2f}\n")
        self.report_text.insert(tk.END, f"Word Count: {word_count}\n")

        self.report_text.insert(tk.END, "\n--- Full Student Test Text ---\n")
        self.report_text.insert(tk.END, student_text)

    def generate_graph(self):
        """Generates a graph based on user selection and class data."""
        class_name = self.class_name_entry.get()
        if not class_name:
            messagebox.showwarning("Missing Class Name", "Please enter a class name to generate a graph.")
            return

        data = get_class_data(class_name)
        if len(data) < 1:
            messagebox.showinfo("No Data", "No data found for this class. Please analyze some papers first.")
            return

        # Clear any previous graph from the dedicated frame
        for widget in self.graph_display_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(8, 6))

        graph_type = self.graph_type.get()
        grades = [row[0] for row in data]
        llm_confidences = [row[1] for row in data]
        readabilities = [row[2] for row in data]
        dates = [datetime.datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S") for row in data]

        if graph_type == "Grade Scores":
            ax.plot(dates, grades, marker='o', linestyle='-', color='b')
            ax.set_xlabel('Date')
            ax.set_ylabel('Grade Score (%)')
            ax.set_title(f"{class_name} Grade Scores Over Time")
            fig.autofmt_xdate()
        elif graph_type == "LLM Confidence":
            ax.plot(dates, llm_confidences, marker='o', linestyle='-', color='g')
            ax.set_xlabel('Date')
            ax.set_ylabel('LLM Confidence (%)')
            ax.set_title(f"{class_name} LLM Confidence Over Time")
            fig.autofmt_xdate()
        elif graph_type == "Grade vs Confidence":
            ax.scatter(grades, llm_confidences, color='purple')
            ax.set_xlabel('Grade Score (%)')
            ax.set_ylabel('LLM Confidence (%)')
            ax.set_title(f"Grade vs LLM Confidence in {class_name}")
        elif graph_type == "Readability vs Confidence":
            ax.scatter(readabilities, llm_confidences, color='red')
            ax.set_xlabel('Readability Score')
            ax.set_ylabel('LLM Confidence (%)')
            ax.set_title(f"Readability vs LLM Confidence in {class_name}")

        canvas = FigureCanvasTkAgg(fig, master=self.graph_display_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def save_report_to_file(self):
        """Saves the contents of the report text box to a .txt or .pdf file."""
        report_content = self.report_text.get("1.0", tk.END)
        if not report_content.strip():
            messagebox.showwarning("No Report", "There is no analysis report to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("PDF Files", "*.pdf"), ("All Files", "*.*")],
            title="Save Analysis Report"
        )

        if file_path:
            try:
                if file_path.lower().endswith('.pdf'):
                    self.save_as_pdf(file_path, report_content)
                else:
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(report_content)
                messagebox.showinfo("Success", f"Report successfully saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")

    def save_as_pdf(self, file_path, text_content):
        """Saves the text content to a PDF file."""
        try:
            c = canvas.Canvas(file_path, pagesize=letter)
            text_object = c.beginText(40, 750)
            text_object.setFont("Helvetica", 10)

            lines = text_content.split('\n')

            for line in lines:
                text_object.textLine(line)
                if text_object.getY() < 50:
                    c.drawText(text_object)
                    c.showPage()
                    text_object = c.beginText(40, 750)
                    text_object.setFont("Helvetica", 10)

            c.drawText(text_object)
            c.save()
        except Exception as e:
            messagebox.showerror("PDF Save Error", f"Failed to save PDF: {e}")


if __name__ == "__main__":
    setup_database()
    app = AutoGraderApp()
    app.mainloop()
