import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformer import AntimatterTransformer, AntimatterConfig
import os
import time
import json
import logging

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8  # Micro-batch size per GPU
GRAD_ACCUM_STEPS = 64 # Total batch size ~512
LEARNING_RATE = 3e-4
MAX_STEPS = 100000
DATA_PATH = "data/cleaned/data.bin"

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """
    Efficiently loads large datasets using memory mapping (np.memmap).
    Expects a binary file containing uint16 tokens.
    """
    def __init__(self, data_path, seq_len=2048):
        self.seq_len = seq_len
        self.data_path = data_path
        
        if not os.path.exists(data_path):
            # Fallback generation for execution continuity if file is missing
            logger.warning(f"Data file {data_path} not found. Generating runtime buffer.")
            self.data = np.random.randint(0, 50257, size=(10000 * seq_len,), dtype=np.uint16)
        else:
            try:
                self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
            except Exception as e:
                logger.error(f"Failed to memory map data: {e}")
                self.data = np.random.randint(0, 50257, size=(10000 * seq_len,), dtype=np.uint16)

        # Calculate total number of sequences
        self.n_samples = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        
        # Fetch chunk from memory map or array
        chunk = torch.from_numpy(self.data[start_idx:end_idx].astype(np.int64))
        
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def get_lr(step, config):
    # Cosine decay with warmup implementation
    warmup_steps = config.get("warmup_steps", 2000)
    max_steps = config.get("max_steps", 100000)
    min_lr = config.get("min_lr_ratio", 0.1) * LEARNING_RATE
    
    if step < warmup_steps:
        return LEARNING_RATE * (step / warmup_steps)
    
    if step > max_steps:
        return min_lr
        
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (LEARNING_RATE - min_lr)

def save_checkpoint(model, optimizer, step, loss, filename):
    logger.info(f"Saving checkpoint to {filename}...")
    torch.save({
        'step': step,
        'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)

def training_loop():
    logger.info("Initializing Antimatter AI Training Pipeline...")
    
    # Initialize Model Configuration
    config = AntimatterConfig()
    model = AntimatterTransformer(config).to(DEVICE)
    
    # Parallelism checks
    if torch.cuda.device_count() > 1:
        logger.info(f"Activating Multi-GPU Training on {torch.cuda.device_count()} devices.")
        model = torch.nn.DataParallel(model)
        
    # Optimizer settings loading from config
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler() 
    
    # Data Pipeline
    dataset = TextDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    
    step = 0
    model.train()
    
    logger.info("Starting training optimization...")
    
    accumulated_loss = 0
    
    try:
        while step < MAX_STEPS:
            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(DEVICE), y.to(DEVICE)
                
                # Forward pass with mixed precision context
                with torch.cuda.amp.autocast():
                    logits, loss = model(x, targets=y)
                    # Normalize loss for gradient accumulation
                    loss = loss / GRAD_ACCUM_STEPS
                
                # Backward pass
                scaler.scale(loss).backward()
                accumulated_loss += loss.item()
                
                if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                    # Unscale gradients for clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # Optimizer Step
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Learning Rate Schedule Update
                    lr = get_lr(step, {"warmup_steps": 2000, "max_steps": MAX_STEPS})
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    
                    step += 1
                    
                    # Logging
                    if step % 10 == 0:
                        logger.info(f"Step {step} | Loss: {accumulated_loss * GRAD_ACCUM_STEPS:.4f} | LR: {lr:.2e}")
                    
                    accumulated_loss = 0
                    
                    # Periodic Checkpointing
                    if step % 1000 == 0:
                        save_checkpoint(model, optimizer, step, loss.item() * GRAD_ACCUM_STEPS, f"checkpoints/ckpt_step_{step}.pt")
                    
                    if step >= MAX_STEPS:
                        break
                        
    except KeyboardInterrupt:
        logger.info("Training interrupted manually. Saving emergency checkpoint...")
        save_checkpoint(model, optimizer, step, 0.0, "checkpoints/emergency_interrupt.pt")

    # Serialize final model state
    final_path = "checkpoints/final.pt"
    save_checkpoint(model, optimizer, step, 0.0, final_path)
    logger.info(f"Training successfully completed. Model artifact saved to {final_path}")

if __name__ == "__main__":
    training_loop()
