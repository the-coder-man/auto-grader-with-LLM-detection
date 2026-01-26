#!/bin/bash

echo "========================================"
echo "     Auto Grader Setup Script"
echo "========================================"
echo ""

OS=""
PACKAGE_MANAGER=""
PYTHON_CMD=""
PIP_CMD=""
USE_VENV=false

detect_os() {
    echo "[1/6] Detecting operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        echo "      Detected: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        echo "      Detected: macOS"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS="windows"
        echo "      Detected: Windows (Git Bash/Cygwin/MSYS)"
    elif [[ -f /etc/os-release ]]; then
        OS="linux"
        echo "      Detected: Linux"
    else
        echo "      Warning: Could not detect OS. Assuming Linux."
        OS="linux"
    fi
    echo ""
}

install_homebrew() {
    echo ""
    echo "      Homebrew is the recommended package manager for macOS."
    read -p "      Would you like to install Homebrew now? (y/n): " brew_choice
    
    if [[ "$brew_choice" == "y" || "$brew_choice" == "Y" ]]; then
        echo ""
        echo "      Installing Xcode Command Line Tools (required for Homebrew)..."
        xcode-select --install 2>/dev/null || true
        
        echo ""
        echo "      Note: If a popup appeared, please complete the Xcode tools installation"
        echo "      and then press Enter to continue..."
        read -p "      Press Enter when ready: "
        
        echo ""
        echo "      Installing Homebrew..."
        echo "      Running: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo ""
        
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        if [[ $? -eq 0 ]]; then
            echo ""
            echo "      Configuring Homebrew PATH..."
            
            ARCH=$(uname -m)
            if [[ "$ARCH" == "arm64" ]]; then
                echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
                eval "$(/opt/homebrew/bin/brew shellenv)"
                echo "      Added Homebrew to PATH for Apple Silicon Mac"
            else
                echo "      Homebrew installed to /usr/local (Intel Mac - PATH already configured)"
            fi
            
            if command -v brew &> /dev/null; then
                PACKAGE_MANAGER="brew"
                echo ""
                echo "      Homebrew installed successfully!"
                return 0
            fi
        fi
        
        echo ""
        echo "      Warning: Homebrew installation may have failed."
        echo "      You can try installing manually: https://brew.sh"
        return 1
    else
        echo "      Skipping Homebrew installation."
        return 1
    fi
}

detect_package_manager() {
    echo "[2/6] Detecting package manager..."
    
    if [[ "$OS" == "macos" ]]; then
        if command -v brew &> /dev/null; then
            PACKAGE_MANAGER="brew"
            echo "      Detected: Homebrew"
        elif command -v port &> /dev/null; then
            PACKAGE_MANAGER="port"
            echo "      Detected: MacPorts"
        else
            echo "      No package manager found on macOS."
            install_homebrew
            if [[ "$PACKAGE_MANAGER" != "brew" ]]; then
                PACKAGE_MANAGER="none"
            fi
        fi
    elif [[ "$OS" == "linux" ]]; then
        if command -v apt-get &> /dev/null; then
            PACKAGE_MANAGER="apt"
            echo "      Detected: APT (Debian/Ubuntu)"
        elif command -v dnf &> /dev/null; then
            PACKAGE_MANAGER="dnf"
            echo "      Detected: DNF (Fedora/RHEL)"
        elif command -v yum &> /dev/null; then
            PACKAGE_MANAGER="yum"
            echo "      Detected: YUM (CentOS/RHEL)"
        elif command -v pacman &> /dev/null; then
            PACKAGE_MANAGER="pacman"
            echo "      Detected: Pacman (Arch Linux)"
        elif command -v zypper &> /dev/null; then
            PACKAGE_MANAGER="zypper"
            echo "      Detected: Zypper (openSUSE)"
        elif command -v apk &> /dev/null; then
            PACKAGE_MANAGER="apk"
            echo "      Detected: APK (Alpine Linux)"
        elif command -v nix-env &> /dev/null; then
            PACKAGE_MANAGER="nix"
            echo "      Detected: Nix"
        else
            echo "      Warning: No supported package manager found."
            PACKAGE_MANAGER="none"
        fi
    elif [[ "$OS" == "windows" ]]; then
        if command -v choco &> /dev/null; then
            PACKAGE_MANAGER="choco"
            echo "      Detected: Chocolatey"
        elif command -v scoop &> /dev/null; then
            PACKAGE_MANAGER="scoop"
            echo "      Detected: Scoop"
        elif command -v winget &> /dev/null; then
            PACKAGE_MANAGER="winget"
            echo "      Detected: Winget"
        else
            echo "      Warning: No package manager found."
            echo "      Please install Chocolatey: https://chocolatey.org"
            PACKAGE_MANAGER="none"
        fi
    fi
    echo ""
}

check_python() {
    echo "[3/6] Checking for Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PIP_CMD="pip3"
        PYTHON_VERSION=$(python3 --version 2>&1)
        echo "      Found: $PYTHON_VERSION"
        return 0
    elif command -v python &> /dev/null; then
        VERSION_CHECK=$(python -c "import sys; print(sys.version_info[0])" 2>/dev/null)
        if [[ "$VERSION_CHECK" == "3" ]]; then
            PYTHON_CMD="python"
            PIP_CMD="pip"
            PYTHON_VERSION=$(python --version 2>&1)
            echo "      Found: $PYTHON_VERSION"
            return 0
        fi
    fi
    
    echo "      Python 3 not found."
    return 1
}

install_python() {
    echo ""
    echo "      Attempting to install Python 3..."
    
    case $PACKAGE_MANAGER in
        apt)
            echo "      Running: sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv python3-tk"
            sudo apt-get update
            sudo apt-get install -y python3 python3-pip python3-venv python3-tk
            ;;
        dnf)
            echo "      Running: sudo dnf install -y python3 python3-pip python3-tkinter"
            sudo dnf install -y python3 python3-pip python3-tkinter
            ;;
        yum)
            echo "      Running: sudo yum install -y python3 python3-pip python3-tkinter"
            sudo yum install -y python3 python3-pip python3-tkinter
            ;;
        pacman)
            echo "      Running: sudo pacman -S --noconfirm python python-pip tk"
            sudo pacman -S --noconfirm python python-pip tk
            ;;
        zypper)
            echo "      Running: sudo zypper install -y python3 python3-pip python3-tk"
            sudo zypper install -y python3 python3-pip python3-tk
            ;;
        apk)
            echo "      Running: sudo apk add python3 py3-pip tk"
            sudo apk add python3 py3-pip tk
            ;;
        nix)
            echo "      Running: nix-env -iA nixpkgs.python3 nixpkgs.python3Packages.pip"
            nix-env -iA nixpkgs.python3 nixpkgs.python3Packages.pip
            ;;
        brew)
            echo "      Running: brew install python python-tk"
            brew install python python-tk
            ;;
        port)
            echo "      Running: sudo port install python311 py311-pip py311-tkinter"
            sudo port install python311 py311-pip py311-tkinter
            ;;
        choco)
            echo "      Running: choco install python -y"
            choco install python -y
            ;;
        scoop)
            echo "      Running: scoop install python"
            scoop install python
            ;;
        winget)
            echo "      Running: winget install Python.Python.3.11"
            winget install Python.Python.3.11
            ;;
        none)
            echo ""
            echo "      ERROR: No package manager available to install Python."
            echo "      Please install Python 3.11+ manually from: https://www.python.org/downloads/"
            exit 1
            ;;
    esac
    
    if ! check_python; then
        echo ""
        echo "      ERROR: Python installation failed."
        echo "      Please install Python 3.11+ manually from: https://www.python.org/downloads/"
        exit 1
    fi
    
    echo "      Python installed successfully!"
}

ask_virtual_environment() {
    echo "[4/6] Virtual Environment Setup"
    echo ""
    read -p "      Would you like to create and activate a Python virtual environment? (y/n): " venv_choice
    
    if [[ "$venv_choice" == "y" || "$venv_choice" == "Y" ]]; then
        USE_VENV=true
        echo ""
        echo "      Creating virtual environment..."
        
        $PYTHON_CMD -m venv venv
        
        if [[ $? -ne 0 ]]; then
            echo "      Warning: Failed to create virtual environment."
            echo "      Continuing without virtual environment..."
            USE_VENV=false
        else
            echo "      Virtual environment created in ./venv"
            echo "      Activating virtual environment..."
            
            if [[ "$OS" == "windows" ]]; then
                source venv/Scripts/activate
            else
                source venv/bin/activate
            fi
            
            PYTHON_CMD="python"
            PIP_CMD="pip"
            echo "      Virtual environment activated!"
        fi
    else
        echo "      Skipping virtual environment setup."
    fi
    echo ""
}

install_dependencies() {
    echo "[5/6] Installing Python dependencies..."
    echo ""
    
    if [[ -f "requirements.txt" ]]; then
        echo "      Found requirements.txt"
        echo "      Running: $PIP_CMD install -r requirements.txt"
        echo ""
        
        $PIP_CMD install -r requirements.txt
        
        if [[ $? -ne 0 ]]; then
            echo ""
            echo "      Warning: Some dependencies may have failed to install."
            echo "      You may need to install them manually."
        else
            echo ""
            echo "      All dependencies installed successfully!"
        fi
    else
        echo "      ERROR: requirements.txt not found!"
        echo "      Please ensure you are running this script from the project directory."
        exit 1
    fi
    echo ""
}

ask_run_application() {
    echo "[6/6] Setup Complete!"
    echo ""
    echo "========================================"
    echo "      Setup completed successfully!"
    echo "========================================"
    echo ""
    
    if [[ "$USE_VENV" == true ]]; then
        echo "Note: A virtual environment was created."
        echo "To activate it in future sessions, run:"
        if [[ "$OS" == "windows" ]]; then
            echo "      source venv/Scripts/activate"
        else
            echo "      source venv/bin/activate"
        fi
        echo ""
    fi
    
    read -p "Would you like to run the Auto Grader application now? (y/n): " run_choice
    
    if [[ "$run_choice" == "y" || "$run_choice" == "Y" ]]; then
        echo ""
        echo "Starting Auto Grader..."
        echo ""
        $PYTHON_CMD "auto grader.py"
    else
        echo ""
        echo "To run the application later, use:"
        if [[ "$USE_VENV" == true ]]; then
            if [[ "$OS" == "windows" ]]; then
                echo "      source venv/Scripts/activate"
            else
                echo "      source venv/bin/activate"
            fi
        fi
        echo "      $PYTHON_CMD \"auto grader.py\""
        echo ""
        echo "Thank you for using Auto Grader!"
    fi
}

main() {
    detect_os
    detect_package_manager
    
    if ! check_python; then
        read -p "      Would you like to install Python now? (y/n): " install_choice
        if [[ "$install_choice" == "y" || "$install_choice" == "Y" ]]; then
            install_python
        else
            echo ""
            echo "      Python 3 is required to run this application."
            echo "      Please install Python 3.11+ and run this script again."
            exit 1
        fi
    fi
    echo ""
    
    ask_virtual_environment
    install_dependencies
    ask_run_application
}

main
