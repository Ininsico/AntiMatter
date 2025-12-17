import os
import subprocess
import time
import requests
import sys
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OllamaPipeline")

OLLAMA_HOST = "http://localhost:11434"
INSTALL_SCRIPT_URL = "https://ollama.com/install.sh"

def check_command(command):
    """Verifies if a command exists in the system path."""
    return subprocess.call(f"which {command}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0

def install_ollama():
    """Identifies OS and attempts installation of Ollama runtime."""
    if check_command("ollama"):
        logger.info("Ollama is already installed.")
        return

    logger.info("Ollama not found. Initiating installation sequence...")
    try:
        # Detect environment (Linux/Mac only for this script scope)
        subprocess.check_call(f"curl -fsSL {INSTALL_SCRIPT_URL} | sh", shell=True)
        logger.info("Ollama installation successful.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Installation failed: {e}")
        sys.exit(1)

def start_server():
    """Starts the Ollama server process in the background."""
    logger.info("Starting Ollama background service...")
    
    try:
        # Launch process ensuring it doesn't block
        process = subprocess.Popen(
            ["ollama", "serve"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid # Detach from parent
        )
        
        # Health check polling
        retries = 10
        for i in range(retries):
            try:
                resp = requests.get(OLLAMA_HOST)
                if resp.status_code == 200:
                    logger.info("Ollama server is active and healthy.")
                    return process
            except requests.ConnectionError:
                pass
            
            logger.debug(f"Waiting for server... ({i+1}/{retries})")
            time.sleep(2)
            
        logger.error("Timed out waiting for Ollama server start.")
        process.kill()
        return None
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return None

def deploy_model(model_name="antimatter", modelfile_path="Modelfile"):
    """Creates the specific model instance within Ollama."""
    logger.info(f"Building model '{model_name}' from {modelfile_path}...")
    
    if not os.path.exists(modelfile_path):
        logger.error(f"Modelfile not found at {modelfile_path}")
        return False

    try:
        subprocess.check_call(["ollama", "create", model_name, "-f", modelfile_path])
        logger.info(f"Model '{model_name}' successfully built and registered.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Model creation failed: {e}")
        return False

def orchestration_pipeline():
    logger.info("=== Starting Antimatter Kaggle Deployment Pipeline ===")
    
    install_ollama()
    
    server_process = start_server()
    if not server_process:
        sys.exit(1)
        
    if deploy_model():
        logger.info("Pipeline Execution Validated. System Ready for Inference.")
        
        # Keep process alive for Kaggle session
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Pipeline terminated by user.")
            os.killpg(os.getpgid(server_process.pid), 15) # Clean shutdown
    else:
        logger.error("Deployment failed.")
        os.killpg(os.getpgid(server_process.pid), 15)

if __name__ == "__main__":
    orchestration_pipeline()
