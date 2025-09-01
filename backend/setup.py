#!/usr/bin/env python3

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔹 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def setup_backend():
    """Setup the backend environment"""
    print("🚀 Setting up AI Browser Automation Backend...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Install Playwright browsers
    if not run_command("playwright install chromium", "Installing Playwright browsers"):
        return False
    
    # Create directories
    os.makedirs("screenshots", exist_ok=True)
    print("✅ Created screenshots directory")
    
    print("\n🎉 Backend setup complete!")
    print("\n📝 Next steps:")
    print("1. Set your GROQ_API_KEY environment variable:")
    print("   export GROQ_API_KEY='your-groq-api-key-here'")
    print("2. Run the backend: python main.py")
    print("3. Backend will be available at: http://localhost:8000")
    print("4. API docs will be at: http://localhost:8000/docs")
    
    return True

if __name__ == "__main__":
    setup_backend()