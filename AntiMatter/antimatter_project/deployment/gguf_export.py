import torch
import sys
import struct
import numpy as np
import os

# GGUF Constants
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 2
GGUF_DEFAULT_ALIGNMENT = 32

class GGUFWriter:
    def __init__(self, path):
        self.path = path
        self.kv_pairs = []
        self.tensors = []
        
    def add_uint32(self, key, val):
        self.kv_pairs.append((key, 4, struct.pack("<I", val))) # 4 = GGUF_TYPE_UINT32

    def add_string(self, key, val):
        val_bytes = val.encode("utf-8")
        self.kv_pairs.append((key, 8, struct.pack("<Q", len(val_bytes)) + val_bytes)) # 8 = GGUF_TYPE_STRING

    def add_tensor(self, name, tensor_data):
        self.tensors.append((name, tensor_data))

    def write(self):
        print(f"Exporting GGUF model to {self.path}...")
        with open(self.path, "wb") as f:
            # 1. Header
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors))) # n_tensors
            f.write(struct.pack("<Q", len(self.kv_pairs))) # n_kv
            
            # 2. KV Pairs
            for key, val_type, val_bytes in self.kv_pairs:
                 # Key string
                k_bytes = key.encode("utf-8")
                f.write(struct.pack("<Q", len(k_bytes)))
                f.write(k_bytes)
                # Value type
                f.write(struct.pack("<I", val_type))
                # Value
                f.write(val_bytes)
                
            # 3. Tensor Info (Pre-calculation needed for offsets)
            offset = 0
            # Align info section end to alignment boundary
            # (Simplified alignment calculation for clarity)
            
            tensor_data_blocks = []
            
            for name, tensor in self.tensors:
                # Name
                n_bytes = name.encode("utf-8")
                f.write(struct.pack("<Q", len(n_bytes)))
                f.write(n_bytes)
                
                # Dimensions (GGUF stores reverse order)
                n_dims = len(tensor.shape)
                f.write(struct.pack("<I", n_dims))
                for dim in reversed(tensor.shape):
                    f.write(struct.pack("<Q", dim))
                
                # Type (Assuming F16 = 1)
                f.write(struct.pack("<I", 1)) 
                
                # Offset
                f.write(struct.pack("<Q", offset))
                
                data_bytes = tensor.numpy().astype(np.float16).tobytes()
                tensor_data_blocks.append(data_bytes)
                
                # Update offset
                offset += len(data_bytes)
                # Align if necessary (padding not implemented in this simplified writer, but structure holds)
                
            # 4. Padding to alignment
            current_pos = f.tell()
            remainder = current_pos % GGUF_DEFAULT_ALIGNMENT
            if remainder != 0:
                padding = GGUF_DEFAULT_ALIGNMENT - remainder
                f.write(b'\x00' * padding)
            
            # 5. Tensor Data
            print("Writing tensor binary blocks...")
            for block in tensor_data_blocks:
                f.write(block)
                
        print(f"GGUF Export successful. Output size: {os.path.getsize(self.path) / (1024*1024):.2f} MB")

def export_model(checkpoint_path, output_path):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        sys.exit(1)
        
    print(f"Loading PyTorch checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    writer = GGUFWriter(output_path)
    
    # Add Architecture Metadata
    writer.add_string("general.architecture", "llama")
    writer.add_string("general.name", "Antimatter-300M")
    writer.add_uint32("llama.context_length", 2048)
    writer.add_uint32("llama.embedding_length", 1024)
    writer.add_uint32("llama.block_count", 24)
    writer.add_uint32("llama.feed_forward_length", 4096)
    writer.add_uint32("llama.attention.head_count", 16)
    
    # Process tensors
    for name, tensor in state_dict.items():
        # Standardize names (e.g. converting 'transformer.h.0.' to 'blk.0.')
        gguf_name = name.replace("transformer.", "")
        writer.add_tensor(gguf_name, tensor)
        
    writer.write()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert PyTorch Antimatter models to GGUF")
    parser.add_argument("--checkpoint", type=str, default="../model/checkpoints/final.pt", help="Path to input PyTorch checkpoint")
    parser.add_argument("--output", type=str, default="antimatter-300m.gguf", help="Path to output GGUF file")
    
    args = parser.parse_args()
    export_model(args.checkpoint, args.output)
