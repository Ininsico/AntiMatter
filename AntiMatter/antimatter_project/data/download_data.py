from datasets import load_dataset
import os
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataDownloader")

DATA_DIR = "data/raw"
OUTPUT_FILE = os.path.join(DATA_DIR, "dataset.txt")

def download_refinedweb(subset_size_gb=45):
    """
    Downloads the Falcon RefinedWeb dataset (clean subset) from Hugging Face.
    """
    logger.info(f"Initializing data download pipeline. Target size: ~{subset_size_gb}GB")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Streaming the dataset to avoid massive RAM usage
    logger.info("Connecting to Hugging Face Hub (tiiuae/falcon-refinedweb)...")
    try:
        dataset = load_dataset("tiiuae/falcon-refinedweb", split="train", streaming=True)
    except Exception as e:
        logger.error(f"Failed to connect to HF Hub: {e}")
        return

    logger.info(f"Streaming data to {OUTPUT_FILE}...")
    
    total_bytes = 0
    target_bytes = subset_size_gb * 1024 * 1024 * 1024
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        progress_bar = tqdm(total=target_bytes, unit='B', unit_scale=True, desc="Downloading")
        
        for sample in dataset:
            content = sample.get('content', '')
            if content:
                # Add newline separator
                text_chunk = content + "\n<|endoftext|>\n"
                f.write(text_chunk)
                
                chunk_size = len(text_chunk.encode('utf-8'))
                total_bytes += chunk_size
                progress_bar.update(chunk_size)
                
                if total_bytes >= target_bytes:
                    break
                    
        progress_bar.close()

    logger.info(f"Download complete. Total size: {total_bytes / (1024**3):.2f} GB")
    logger.info("Verifying file integrity...")
    if os.path.getsize(OUTPUT_FILE) > 0:
        logger.info("Integrity check passed.")
    else:
        logger.error("File is empty. Download failed.")

if __name__ == "__main__":
    download_refinedweb()
