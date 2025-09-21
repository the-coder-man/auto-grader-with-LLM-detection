# **Getting Started: Installing Python and Libraries**

This guide will walk you through the process of setting up your computer to run the auto grader.py program. It is designed for beginners who do not yet have Python installed.

## **Step 1: Install Python**

First, you need to install Python. This will install the core language and a tool called pip, which we will use to install the required libraries.

1. **Go to the official Python website:** Open your web browser and navigate to

 [https://www.python.org/downloads/](https://www.python.org/downloads/)

2. **Download the installer:** The website should automatically detect your operating system (Windows, macOS, etc.) and recommend the correct version. Click the button to download the latest stable version of Python.  
3. **Run the installer:**  
   * **For Windows:** Double-click the downloaded .exe file. **IMPORTANT:** On the first screen of the installer, **make sure to check the box that says "Add Python to PATH"** before clicking "Install Now". This is a crucial step that makes it easy to run Python from your command line.  **For macOS:** Double-click the downloaded .pkg file and follow the on-screen instructions.

## **Step 2: Open a Command Line or Terminal**

Now that Python is installed, you need to open a command line tool to run commands.

* **On Windows:** Press the Windows key, type cmd or Command Prompt, and press Enter.  
* **On macOS:** Press Cmd \+ Space to open Spotlight Search, type Terminal, and press Enter.

## **Step 3: Navigate to Your Project Folder**

You need to tell the command line where your project files (auto grader.py and requirements.txt) are located.

1. In the command line, type cd (which means "change directory") followed by a space.  
2. Drag and drop your project folder directly into the command line window. This will automatically paste the correct path.  
3. Press Enter. You should now see that your command line's location has changed to your project folder.

## **Step 4: Install the Required Libraries**

The requirements.txt file lists all the Python libraries that your auto grader.py file needs to work. We'll use pip to install them all at once.

1. In your command line, while still in your project folder, type the following command and press Enter:  
   pip install \-r requirements.txt

   * pip install: This is the command to install packages.  
   * \-r: This stands for "requirements" and tells pip to read from a file.  
   * requirements.txt: This is the name of the file containing the list of libraries.  
2. pip will now download and install all the necessary libraries, including: fitz, scikit-learn, numpy, matplotlib, spacy, pandas, and reportlab. You will see a lot of text scroll by as it installs each one.  
3. **Optional, but recommended:** After this, you will also need to download the en\_core\_web\_sm spaCy model, which your script uses. Run this command:  
   python \-m spacy download en\_core\_web\_sm

That's it\! You have now successfully set up your environment to run the auto grader.py program. You can now run the Python file by typing python "auto grader.py" in the same command prompt window.