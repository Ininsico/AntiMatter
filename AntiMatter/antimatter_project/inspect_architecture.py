import torch
from model.transformer import AntimatterTransformer, AntimatterConfig
import os

def inspect_model():
    print("Loading Antimatter-300M Architecture...")
    config = AntimatterConfig()
    model = AntimatterTransformer(config)
    
    # Print Model Architecture (Visual Proof)
    print("\n" + "="*50)
    print("ANTIMATTER AI - MODEL SUMMARY")
    print("="*50)
    print(model)
    print("="*50)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Trainable Parameters: {total_params:,}")
    print(f"Model Configuration: Layers={config.num_layers}, Hidden={config.hidden_size}, Heads={config.num_heads}")
    
    # Check for Checkpoints
    ckpt_path = "model/checkpoints/final.pt"
    if os.path.exists(ckpt_path):
        print(f"\nVerifying Checkpoint at {ckpt_path}...")
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            print(f"Checkpoint loaded successfully.")
            print(f"Iterations: {ckpt.get('step', 'Unknown')}")
            print(f"Final Loss: {ckpt.get('loss', 'Unknown')}")
            print(f"Optimizer keys: {list(ckpt.get('optimizer_state_dict', {}).keys())}")
        except Exception as e:
            print(f"Checkpoint verification failed: {e}")
    else:
        print("\nNo checkpoint found to verify.")

if __name__ == "__main__":
    inspect_model()
