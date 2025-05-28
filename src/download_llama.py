import os
import sys
from huggingface_hub import snapshot_download

def main():
    # Add SkinGPT-4 directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    skingpt_dir = os.path.join(os.path.dirname(current_dir), 'SkinGPT-4')
    sys.path.append(skingpt_dir)
    
    # Create weights directory if it doesn't exist
    weights_dir = os.path.join(skingpt_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    # Download Llama model files
    print("Downloading Llama-2-13b-chat-hf model files...")
    model_path = os.path.join(weights_dir, "Llama-2-13b-chat-hf")
    snapshot_download(
        repo_id="meta-llama/Llama-2-13b-chat-hf",
        local_dir=model_path,
        token=os.environ.get("HF_TOKEN"),  # Get token from environment variable
        ignore_patterns=["*.safetensors"],  # Skip safetensors files to save space
    )
    print(f"Downloaded model files to {model_path}")

if __name__ == "__main__":
    main() 