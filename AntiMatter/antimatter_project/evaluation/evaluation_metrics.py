import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Evaluator")

def calculate_perplexity(loss):
    """
    Calculate perplexity from cross-entropy loss.
    PPL = exp(CrossEntropy)
    """
    return math.exp(loss)

def evaluate_model(model, dataloader, device):
    """
    Performs a full pass over the validation set to compute Loss and Perplexity.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    logger.info("Starting evaluation on validation set...")
    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")
    
    with torch.no_grad():
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            # Note: Model forward returns logits, loss if targets provided
            _, loss = model(x, targets=y)
            
            # Weighted accumulation of loss
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_tokens += batch_size
            
            # Update progress bar
            progress_bar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
            
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    ppl = calculate_perplexity(avg_loss)
    
    logger.info(f"Evaluation Complete. Validation Loss: {avg_loss:.4f} | Perplexity: {ppl:.2f}")
    
    return avg_loss, ppl

def coherence_check(model, tokenizer, prompts, device, max_new_tokens=50):
    """
    Qualitative check: Generates text for a list of prompts to verify basic coherence.
    """
    model.eval()
    logger.info("\n=== Coherence & Factuality Check ===")
    
    for prompt in prompts:
        # Encode prompt
        input_ids = torch.tensor(tokenizer.encode(prompt).ids).unsqueeze(0).to(device)
        
        # specific generation loop
        generated_ids = input_ids
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = model(generated_ids)
                last_token_logits = logits[:, -1, :]
                probs = F.softmax(last_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                if next_token.item() == tokenizer.token_to_id("[EOS]"):
                    break
        
        output_text = tokenizer.decode(generated_ids[0].tolist())
        print(f"\n[PROMPT]: {prompt}")
        print(f"[OUTPUT]: {output_text}")

if __name__ == "__main__":
    # Example usage requires existing model and dataloader
    pass
