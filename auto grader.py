import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import PyPDF2
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
import threading
import queue
# Prefer certifi CA bundle for SSL verification (helps fix CERTIFICATE_VERIFY_FAILED on macOS/custom Pythons)
try:
    import certifi
    _CERTIFI_CA = certifi.where()
    # Ensure subprocesses and requests see the CA bundle as well
    os.environ.setdefault('SSL_CERT_FILE', _CERTIFI_CA)
    os.environ.setdefault('REQUESTS_CA_BUNDLE', _CERTIFI_CA)
except Exception:
    _CERTIFI_CA = None

import nltk
nltk.download('punkt')

MODEL_PATH = "llm_model.joblib"
VECTORIZER_PATH = "llm_vectorizer.joblib"

class AutoGraderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Grader")

        self.llm_model = None
        self.vectorizer = None
        self.grading_method = tk.StringVar(value="keyword")
        self.smollm_model = None
        self.smollm_tokenizer = None

        self.text_input = scrolledtext.ScrolledText(root, height=10, width=60)
        self.text_input.pack(pady=5)

        self.pdf_button = tk.Button(root, text="Upload Assignment PDF", command=self.upload_pdf, bg="white", fg="black")
        self.pdf_button.pack(pady=5)

        self.grading_method_frame = tk.Frame(root, relief=tk.GROOVE, borderwidth=2)
        self.grading_method_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(self.grading_method_frame, text="Select Grading Method:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=5)
        
        self.keyword_radio = tk.Radiobutton(self.grading_method_frame, text="Keyword-based Grading (Type keywords and points)", 
                                            variable=self.grading_method, value="keyword", font=("Arial", 9), 
                                            command=self.update_grading_inputs)
        self.keyword_radio.pack(anchor=tk.W, padx=20, pady=2)
        
        self.ai_radio = tk.Radiobutton(self.grading_method_frame, text="AI-Assisted Grading (SmolLM2 - Quality assessment)", 
                                       variable=self.grading_method, value="ai", font=("Arial", 9), 
                                       command=self.update_grading_inputs)
        self.ai_radio.pack(anchor=tk.W, padx=20, pady=2)

        self.keyword_input_frame = tk.Frame(root)
        
        tk.Label(self.keyword_input_frame, text="Keywords and Points:", font=("Arial", 9, "bold")).pack(anchor=tk.W, padx=5)
        tk.Label(self.keyword_input_frame, text="Format: keyword1:points, keyword2:points (e.g., photosynthesis:10, chlorophyll:5)", font=("Arial", 8), fg="gray").pack(anchor=tk.W, padx=5)
        
        self.keyword_input = scrolledtext.ScrolledText(self.keyword_input_frame, height=3, width=60)
        self.keyword_input.pack(pady=5, padx=5)

        self.ai_notes_frame = tk.Frame(root)
        
        tk.Label(self.ai_notes_frame, text="AI Grading Instructions:", font=("Arial", 9, "bold")).pack(anchor=tk.W, padx=5)
        tk.Label(self.ai_notes_frame, text="Describe what you want the AI to evaluate and how to score (e.g., max 100 points):", font=("Arial", 8), fg="gray").pack(anchor=tk.W, padx=5)
        
        self.ai_notes_input = scrolledtext.ScrolledText(self.ai_notes_frame, height=4, width=60)
        self.ai_notes_input.pack(pady=5, padx=5)

        self.grade_button = tk.Button(root, text="Grade Assignment", command=self.grade_assignment, bg="white", fg="black")
        self.grade_button.pack(pady=5)

        self.llm_detect_button = tk.Button(root, text="Detect AI-Generated Text", command=self.detect_llm_text, bg="white", fg="black")
        self.llm_detect_button.pack(pady=5)

        self.disclaimer_label = tk.Label(root, text="DISCLAIMER: AI detection is not foolproof! This tool provides an estimate based on patterns in the training data, but it can make mistakes. Factors like writing style, topic, and text length can affect accuracy. Use this as one tool among many when evaluating content authenticity.", 
                                         fg="#FF6B6B", font=("Arial", 9, "bold"), wraplength=400)
        self.disclaimer_label.pack(pady=2)

        self.train_button = tk.Button(root, text="Train AI Model", command=self.train_llm_model, bg="white", fg="black")
        self.train_button.pack(pady=5)

        self.training_status_label = tk.Label(root, text="", fg="blue")
        self.training_status_label.pack(pady=2)

        self.progress_bar = ttk.Progressbar(root, length=400, mode='determinate')
        self.progress_bar.pack(pady=5)

        self.result_label = tk.Label(root, text="Total Score: N/A")
        self.result_label.pack(pady=5)

        self.scores = []

        self.progress_queue = queue.Queue()
        self.is_training = False

        # Auto-load model if it exists
        self.load_llm_model()
        self.update_train_button_text()
        
        # Initialize with keyword input visible by default
        self.keyword_input_frame.pack(pady=5, padx=20, fill=tk.X, before=self.grade_button)
        
        # Load SmolLM2 model on startup in background
        self.load_smollm_on_startup()

    def update_grading_inputs(self):
        if self.grading_method.get() == "ai":
            self.keyword_input_frame.pack_forget()
            self.ai_notes_frame.pack(pady=5, padx=20, fill=tk.X, before=self.grade_button)
        else:
            self.ai_notes_frame.pack_forget()
            self.keyword_input_frame.pack(pady=5, padx=20, fill=tk.X, before=self.grade_button)

    def upload_pdf(self):
        pdf_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if pdf_path:
            text = self.extract_text_from_pdf(pdf_path)
            self.text_input.delete("1.0", tk.END)
            self.text_input.insert(tk.END, text)

    def extract_text_from_pdf(self, path):
        try:
            with open(path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            messagebox.showerror("PDF Error", f"Failed to extract text: {e}")
            return ""


    def grade_assignment(self):
        student_text = self.text_input.get("1.0", tk.END).strip()
        if not student_text:
            messagebox.showwarning("Warning", "Please upload or type the assignment text.")
            return

        grading_method = self.grading_method.get()
        
        if grading_method == "keyword":
            keywords_input = self.keyword_input.get("1.0", tk.END).strip()
            if not keywords_input:
                messagebox.showwarning("Warning", "Please enter keywords and points for grading.")
                return
            
            # Show grading indicator
            self.training_status_label.config(text="Grading assignment...", fg="blue")
            self.grade_button.config(state=tk.DISABLED)
            self.root.update_idletasks()
            
            # Run keyword grading in thread
            thread = threading.Thread(target=self.grade_with_keywords_thread, args=(student_text, keywords_input))
            thread.daemon = True
            thread.start()
            
        elif grading_method == "ai":
            ai_instructions = self.ai_notes_input.get("1.0", tk.END).strip()
            if not ai_instructions:
                messagebox.showwarning("Warning", "Please enter AI grading instructions.")
                return
            
            # Show grading indicator
            self.training_status_label.config(text="AI grading in progress...", fg="blue")
            self.grade_button.config(state=tk.DISABLED)
            self.root.update_idletasks()
            
            # Run AI grading in thread
            thread = threading.Thread(target=self.grade_with_ai_thread, args=(student_text, ai_instructions))
            thread.daemon = True
            thread.start()

    def grade_with_keywords_thread(self, student_text, keywords_input):
        """Thread worker for keyword grading"""
        try:
            response = student_text.lower()
            total_score = 0
            max_score = 0
            individual_scores = []
            
            keyword_pairs = [pair.strip() for pair in keywords_input.split(',')]
            
            for pair in keyword_pairs:
                if ':' not in pair:
                    self.root.after(0, lambda: messagebox.showerror("Format Error", f"Invalid format: '{pair}'. Use keyword:points format."))
                    self.root.after(0, self.finish_grading, None)
                    return
                
                keyword, points_str = pair.split(':', 1)
                keyword = keyword.strip().lower()
                
                try:
                    points = float(points_str.strip())
                except ValueError:
                    self.root.after(0, lambda: messagebox.showerror("Format Error", f"Invalid points value: '{points_str}'. Must be a number."))
                    self.root.after(0, self.finish_grading, None)
                    return
                
                max_score += points
                score = points if keyword in response else 0
                individual_scores.append((keyword, score))
                total_score += score
            
            grade_percentage = (total_score / max_score) * 100 if max_score > 0 else 0
            letter_grade = self.calculate_letter_grade(grade_percentage)
            
            result = {
                'total_score': total_score,
                'max_score': max_score,
                'letter_grade': letter_grade,
                'percentage': grade_percentage,
                'scores': individual_scores,
                'student_text': student_text,
                'keywords_input': keywords_input
            }
            
            self.root.after(0, self.finish_grading, result)
            
        except Exception:
            self.root.after(0, lambda: messagebox.showerror("Parsing Error", f"Error parsing keywords: {e}"))
            self.root.after(0, self.finish_grading, None)

    def grade_with_keywords(self, student_text, keywords_input):
        response = student_text.lower()
        total_score = 0
        max_score = 0
        individual_scores = []
        
        try:
            keyword_pairs = [pair.strip() for pair in keywords_input.split(',')]
            
            for pair in keyword_pairs:
                if ':' not in pair:
                    messagebox.showerror("Format Error", f"Invalid format: '{pair}'. Use keyword:points format.")
                    return
                
                keyword, points_str = pair.split(':', 1)
                keyword = keyword.strip().lower()
                
                try:
                    points = float(points_str.strip())
                except ValueError:
                    messagebox.showerror("Format Error", f"Invalid points value: '{points_str}'. Must be a number.")
                    return
                
                max_score += points
                score = points if keyword in response else 0
                individual_scores.append((keyword, score))
                total_score += score
        
        except Exception as e:
            messagebox.showerror("Parsing Error", f"Error parsing keywords: {e}")
            return

        grade_percentage = (total_score / max_score) * 100 if max_score > 0 else 0
        letter_grade = self.calculate_letter_grade(grade_percentage)

        self.result_label.config(text=f"Total Score: {total_score}/{max_score} | Grade: {letter_grade} ({grade_percentage:.1f}%)")
        self.scores = individual_scores
        self.result_label.update_idletasks()

    def grade_with_ai_thread(self, student_text, ai_instructions):
        """Thread worker for AI grading"""
        try:
            score, ai_response = self.ai_assisted_grading(student_text, ai_instructions)
            
            if score is None:
                self.root.after(0, lambda: self.training_status_label.config(text="AI grading failed", fg="red"))
                self.root.after(0, self.finish_grading, None)
                return

            percentage = min(100, max(0, score))
            letter_grade = self.score_to_letter_grade(percentage)

            result = {
                'ai_score': score,
                'ai_percentage': percentage,
                'ai_letter_grade': letter_grade,
                'scores': [("AI Assessment", score)],
                'ai_response': ai_response,
                'student_text': student_text,
                'ai_instructions': ai_instructions
            }
            
            self.root.after(0, self.finish_grading, result)
            
        except Exception:
            self.root.after(0, lambda: messagebox.showerror("AI Grading Error", f"Error during AI grading: {e}"))
            self.root.after(0, self.finish_grading, None)

    def finish_grading(self, result):
        """Update UI after grading completes"""
        self.grade_button.config(state=tk.NORMAL)
        
        if result is None:
            self.training_status_label.config(text="Grading failed", fg="red")
            return
        
        if 'ai_score' in result:
            # AI grading result
            letter_grade = result.get('ai_letter_grade', 'N/A')
            percentage = result.get('ai_percentage', 0)
            self.result_label.config(text=f"AI Grade: {letter_grade} ({percentage:.1f}%)")
            self.scores = result['scores']
            self.training_status_label.config(text="AI grading completed!", fg="green")
            
            # Show messagebox with result
            messagebox.showinfo("AI Grading Complete", f"Grade: {letter_grade} ({percentage:.1f}%)\n\nThe detailed report is now available.")
            
            # Open report window
            self.show_ai_report(result)
        else:
            # Keyword grading result
            self.result_label.config(text=f"Total Score: {result['total_score']}/{result['max_score']} | Grade: {result['letter_grade']} ({result['percentage']:.1f}%)")
            self.scores = result['scores']
            self.training_status_label.config(text="Grading completed!", fg="green")
            
            # Show messagebox with result
            messagebox.showinfo("Grading Complete", f"Score: {result['total_score']}/{result['max_score']}\nGrade: {result['letter_grade']} ({result['percentage']:.1f}%)\n\nThe detailed report is now available.")
            
            # Open report window
            self.show_keyword_report(result)

    def grade_with_ai(self, student_text, ai_instructions):
        self.training_status_label.config(text="AI grading in progress...", fg="blue")
        self.root.update_idletasks()

        score = self.ai_assisted_grading(student_text, ai_instructions)
        
        if score is None:
            self.training_status_label.config(text="AI grading failed", fg="red")
            return

        self.result_label.config(text=f"AI Grade: {score}")
        self.scores = [("AI Assessment", score)]
        self.training_status_label.config(text="AI grading completed!", fg="green")
        self.result_label.update_idletasks()
    def calculate_letter_grade(self, percentage):
        if percentage >= 90:
            return "A"
        elif percentage >= 80:
            return "B"
        elif percentage >= 70:
            return "C"
        elif percentage >= 60:
            return "D"
        else:
            return "F"

    def train_llm_model(self):
        if self.is_training:
            messagebox.showwarning("Training in Progress", "Model training is already in progress. Please wait.")
            return

        self.is_training = True
        self.train_button.config(state='disabled')
        self.progress_bar['value'] = 0
        self.training_status_label.config(text="Initializing training...")

        training_thread = threading.Thread(target=self._train_worker, daemon=True)
        training_thread.start()

        self._check_progress()

    def _train_worker(self):
        try:
            self.progress_queue.put(("status", "Loading dataset from Hugging Face..."))
            self.progress_queue.put(("progress", 5))

            ds = load_dataset("ahmadreza13/human-vs-Ai-generated-dataset")
            train_data = ds['train']
            total_rows = len(train_data)

            self.progress_queue.put(("status", f"Dataset loaded: {total_rows:,} examples"))
            self.progress_queue.put(("progress", 15))

            # Sample dataset for efficiency (use 500k samples for faster training)
            sample_size = min(500000, total_rows)
            self.progress_queue.put(("status", f"Sampling {sample_size:,} examples for training..."))

            import random
            random.seed(42)
            indices = random.sample(range(total_rows), sample_size)
            sampled_data = train_data.select(indices)

            X = sampled_data['data']
            y = sampled_data['generated']

            self.progress_queue.put(("status", "Building vocabulary with TF-IDF..."))
            self.progress_queue.put(("progress", 25))

            vectorizer = TfidfVectorizer(max_features=5000)
            X_vec = vectorizer.fit_transform(X)

            self.progress_queue.put(("status", "Training classifier..."))
            self.progress_queue.put(("progress", 50))

            model = LogisticRegression(max_iter=1000, verbose=0)
            model.fit(X_vec, y)

            self.progress_queue.put(("status", "Saving model..."))
            self.progress_queue.put(("progress", 90))

            joblib.dump(model, MODEL_PATH)
            joblib.dump(vectorizer, VECTORIZER_PATH)

            self.vectorizer = vectorizer
            self.llm_model = model

            self.progress_queue.put(("progress", 100))
            self.progress_queue.put(("status", f"Training complete! Model trained on {sample_size:,} examples."))
            self.progress_queue.put(("done", "success"))

        except Exception as e:
            self.progress_queue.put(("error", str(e)))

    def _check_progress(self):
        # Drain all available messages from the queue
        while True:
            try:
                msg_type, msg_data = self.progress_queue.get_nowait()

                if msg_type == "status":
                    self.training_status_label.config(text=msg_data, fg="blue")
                elif msg_type == "progress":
                    self.progress_bar['value'] = msg_data
                elif msg_type == "done":
                    self.is_training = False
                    self.train_button.config(state='normal')
                    self.training_status_label.config(text="Training complete!", fg="green")
                    self.update_train_button_text()
                    messagebox.showinfo("Training Complete", "LLM detection model trained and saved successfully!")
                    return
                elif msg_type == "error":
                    self.is_training = False
                    self.train_button.config(state='normal')
                    self.training_status_label.config(text="Training failed", fg="red")
                    messagebox.showerror("Training Error", f"Failed to train model: {msg_data}")
                    return
            except queue.Empty:
                break

        # Continue polling if training is still in progress
        if self.is_training:
            self.root.after(100, self._check_progress)

    def load_llm_model(self):
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
                self.llm_model = joblib.load(MODEL_PATH)
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                self.training_status_label.config(text="AI model loaded successfully!", fg="green")
                return True
            return False
        except Exception as e:
            messagebox.showwarning("Load Warning", f"Could not load saved LLM model: {e}")
            return False

    def update_train_button_text(self):
        if self.llm_model is not None and self.vectorizer is not None:
            self.train_button.config(text="Retrain AI Model (Optional)")
        else:
            self.train_button.config(text="Train AI Model (Required)")

    def load_smollm_on_startup(self):
        """Load SmolLM2 model in background thread on startup"""
        thread = threading.Thread(target=self.load_smollm_model_thread)
        thread.daemon = True
        thread.start()
    
    def load_smollm_model_thread(self):
        """Thread worker to load SmolLM2 model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
            device = "cpu"
            
            self.root.after(0, lambda: self.training_status_label.config(text="Loading SmolLM2 model...", fg="blue"))
            
            self.smollm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.smollm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32
            ).to(device)
            self.smollm_model.eval()
            
            self.root.after(0, lambda: self.training_status_label.config(text="SmolLM2 model loaded successfully!", fg="green"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Model Load Error", f"Failed to load SmolLM2 model: {e}"))
            self.root.after(0, lambda: self.training_status_label.config(text="Failed to load SmolLM2 model", fg="red"))

    def load_smollm_model(self):
        if self.smollm_model is not None and self.smollm_tokenizer is not None:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
            device = "cpu"
            
            self.training_status_label.config(text="Loading SmolLM2 model...", fg="blue")
            self.root.update_idletasks()
            
            self.smollm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.smollm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32
            ).to(device)
            self.smollm_model.eval()
            
            self.training_status_label.config(text="SmolLM2 model loaded successfully!", fg="green")
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load SmolLM2 model: {e}")
            self.training_status_label.config(text="Failed to load SmolLM2 model", fg="red")

    def ai_assisted_grading(self, student_text, ai_instructions):
        if self.smollm_model is None or self.smollm_tokenizer is None:
            self.load_smollm_model()
        
        if self.smollm_model is None or self.smollm_tokenizer is None:
            messagebox.showerror("Error", "SmolLM2 model not loaded. Cannot perform AI grading.")
            return None
        
        import torch
        
        prompt = f"""You are an educational assessment assistant. Evaluate the following student response based on the grading instructions provided.

Grading Instructions:
{ai_instructions}

Student Response:
{student_text}

Based on the student's response and the grading instructions above, provide a score out of 100 and justification.

Provide your assessment in this exact format:
Score: [number out of 100]
Justification: [brief explanation]"""

        messages = [{"role": "user", "content": prompt}]
        input_text = self.smollm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.smollm_tokenizer.encode(input_text, return_tensors="pt").to("cpu")
        
        with torch.no_grad():
            outputs = self.smollm_model.generate(
                inputs,
                max_new_tokens=150,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.smollm_tokenizer.eos_token_id
            )
        
        generated_tokens = outputs[0][inputs.shape[1]:]
        response = self.smollm_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        score = self.parse_ai_score(response)
        return score, response

    def parse_ai_score(self, response):
        import re
        
        score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
            return max(0, score)
        
        number_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:out of|/)\s*\d+', response)
        if number_match:
            score = float(number_match.group(1))
            return max(0, score)
        
        return 0
    
    def score_to_letter_grade(self, percentage):
        """Convert percentage to letter grade"""
        if percentage >= 93:
            return 'A'
        elif percentage >= 90:
            return 'A-'
        elif percentage >= 87:
            return 'B+'
        elif percentage >= 83:
            return 'B'
        elif percentage >= 80:
            return 'B-'
        elif percentage >= 77:
            return 'C+'
        elif percentage >= 73:
            return 'C'
        elif percentage >= 70:
            return 'C-'
        elif percentage >= 67:
            return 'D+'
        elif percentage >= 63:
            return 'D'
        elif percentage >= 60:
            return 'D-'
        else:
            return 'F'

    def show_ai_report(self, result):
        """Display AI grading report in a new window with download options"""
        import datetime
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import inch
        
        report_window = tk.Toplevel(self.root)
        report_window.title("AI Grading Report")
        report_window.geometry("700x600")
        
        # Report text widget
        report_text = scrolledtext.ScrolledText(report_window, wrap=tk.WORD, font=("Arial", 11), padx=15, pady=15)
        report_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Generate report content
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        letter_grade = result.get('ai_letter_grade', 'N/A')
        percentage = result.get('ai_percentage', result.get('ai_score', 0))
        report_content = f"""AI GRADING REPORT
{'='*60}

Date: {timestamp}
Grade: {letter_grade} ({percentage:.1f}%)
Score: {result['ai_score']}/100

GRADING INSTRUCTIONS:
{result['ai_instructions']}

{'='*60}

STUDENT RESPONSE:
{result['student_text']}

{'='*60}

AI EVALUATION:
{result['ai_response']}

{'='*60}
End of Report
"""
        
        report_text.insert(tk.END, report_content)
        report_text.config(state=tk.DISABLED)  # Make read-only
        
        # Button frame
        button_frame = tk.Frame(report_window)
        button_frame.pack(pady=10)
        
        # Download as TXT button
        def download_txt():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
                initialfile=f"AI_Grading_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            if file_path:
                try:
                    with open(file_path, 'w') as f:
                        f.write(report_content)
                    messagebox.showinfo("Success", f"Report saved as TXT:\n{file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save TXT file: {e}")
        
        # Download as PDF button
        def download_pdf():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
                initialfile=f"AI_Grading_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            if file_path:
                try:
                    # Create PDF
                    doc = SimpleDocTemplate(file_path, pagesize=letter)
                    styles = getSampleStyleSheet()
                    story = []
                    
                    # Add title
                    title = Paragraph("<b>AI GRADING REPORT</b>", styles['Title'])
                    story.append(title)
                    story.append(Spacer(1, 0.2*inch))
                    
                    # Add content
                    content_lines = report_content.split('\n')
                    for line in content_lines:
                        if line.strip():
                            para = Paragraph(line.replace('<', '&lt;').replace('>', '&gt;'), styles['Normal'])
                            story.append(para)
                        else:
                            story.append(Spacer(1, 0.1*inch))
                    
                    doc.build(story)
                    messagebox.showinfo("Success", f"Report saved as PDF:\n{file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save PDF file: {e}")
        
        tk.Button(button_frame, text="Download as TXT", command=download_txt, bg="white", fg="black", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Download as PDF", command=download_pdf, bg="white", fg="black", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Close", command=report_window.destroy, bg="white", fg="black", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

    def show_keyword_report(self, result):
        """Display keyword grading report in a new window with download options"""
        import datetime
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import inch
        
        report_window = tk.Toplevel(self.root)
        report_window.title("Keyword Grading Report")
        report_window.geometry("700x600")
        
        # Report text widget
        report_text = scrolledtext.ScrolledText(report_window, wrap=tk.WORD, font=("Arial", 11), padx=15, pady=15)
        report_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Generate report content
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build keyword breakdown
        keyword_breakdown = ""
        for keyword, score in result['scores']:
            status = "✓ Found" if score > 0 else "✗ Not found"
            keyword_breakdown += f"  • {keyword}: {score} points - {status}\n"
        
        report_content = f"""KEYWORD GRADING REPORT
{'='*60}

Date: {timestamp}
Total Score: {result['total_score']}/{result['max_score']}
Letter Grade: {result['letter_grade']}
Percentage: {result['percentage']:.1f}%

KEYWORDS USED:
{result['keywords_input']}

{'='*60}

KEYWORD BREAKDOWN:
{keyword_breakdown}
{'='*60}

STUDENT RESPONSE:
{result['student_text']}

{'='*60}
End of Report
"""
        
        report_text.insert(tk.END, report_content)
        report_text.config(state=tk.DISABLED)  # Make read-only
        
        # Button frame
        button_frame = tk.Frame(report_window)
        button_frame.pack(pady=10)
        
        # Download as TXT button
        def download_txt():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
                initialfile=f"Keyword_Grading_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            if file_path:
                try:
                    with open(file_path, 'w') as f:
                        f.write(report_content)
                    messagebox.showinfo("Success", f"Report saved as TXT:\n{file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save TXT file: {e}")
        
        # Download as PDF button
        def download_pdf():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
                initialfile=f"Keyword_Grading_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            if file_path:
                try:
                    # Create PDF
                    doc = SimpleDocTemplate(file_path, pagesize=letter)
                    styles = getSampleStyleSheet()
                    story = []
                    
                    # Add title
                    title = Paragraph("<b>KEYWORD GRADING REPORT</b>", styles['Title'])
                    story.append(title)
                    story.append(Spacer(1, 0.2*inch))
                    
                    # Add content
                    content_lines = report_content.split('\n')
                    for line in content_lines:
                        if line.strip():
                            para = Paragraph(line.replace('<', '&lt;').replace('>', '&gt;'), styles['Normal'])
                            story.append(para)
                        else:
                            story.append(Spacer(1, 0.1*inch))
                    
                    doc.build(story)
                    messagebox.showinfo("Success", f"Report saved as PDF:\n{file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save PDF file: {e}")
        
        tk.Button(button_frame, text="Download as TXT", command=download_txt, bg="white", fg="black", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Download as PDF", command=download_pdf, bg="white", fg="black", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Close", command=report_window.destroy, bg="white", fg="black", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

    def detect_llm_text(self):
        if self.llm_model is None or self.vectorizer is None:
            messagebox.showwarning("Warning", "Please train the LLM detection model first.")
            return

        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please input or upload text for detection.")
            return

        try:
            # Preprocess text
            text = text.lower()

            X_input = self.vectorizer.transform([text])
            prediction_proba = self.llm_model.predict_proba(X_input)[0]

            prediction = self.llm_model.predict(X_input)[0]
            label = "AI-Generated" if prediction == 1 else "Human-Written"

            # Provide probability score
            ai_prob = prediction_proba[1] if prediction == 1 else prediction_proba[0]
            messagebox.showinfo("Detection Result", 
                f"The text appears to be: {label} with a confidence of {ai_prob:.2%}\n\n"
                f"⚠️ Note: AI detection is not foolproof and can make mistakes. "
                f"Use this result as guidance, not absolute proof.")
        except Exception as e:
            messagebox.showerror("Detection Error", f"Detection failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoGraderApp(root)
    root.mainloop()
