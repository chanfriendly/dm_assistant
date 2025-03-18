import os
import subprocess
import argparse
import platform

def setup_oumi_environment():
    """Set up the environment for Oumi training."""
    # Create virtual environment if it doesn't exist
    if not os.path.exists("dnd_env"):
        print("Creating virtual environment...")
        subprocess.run(["python", "-m", "venv", "dnd_env"])
    
    # Determine activation command based on platform
    if os.name == 'nt':  # Windows
        activate_cmd = os.path.join("dnd_env", "Scripts", "activate")
    else:  # Unix/Linux/Mac
        activate_cmd = f"source {os.path.join('dnd_env', 'bin', 'activate')}"
    
    # Install required packages for Oumi
    print("Installing Oumi and required packages...")
    packages = [
        "torch>=2.2.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "peft>=0.6.0",
        "accelerate>=0.25.0",
        "oumi"  # The main package we want to install
    ]
    
    # We need to run this in shell with the activated environment
    install_cmd = f"{activate_cmd} && pip install {' '.join(packages)}"
    
    if os.name == 'nt':  # Windows
        subprocess.run(install_cmd, shell=True)
    else:  # Unix/Linux/Mac
        subprocess.run(install_cmd, shell=True, executable="/bin/bash")
    
    print("Environment setup complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up Oumi training environment.')
    parser.add_argument('--cuda', action='store_true', help='Install CUDA-enabled version of PyTorch')
    
    args = parser.parse_args()
    setup_oumi_environment()