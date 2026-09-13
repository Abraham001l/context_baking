import os
from huggingface_hub import snapshot_download

def download_qwen_offline():
    model_id = "Qwen/Qwen2.5-1.5B"
    local_path = "./models/Qwen2.5-1.5B_base"
    
    os.makedirs(local_path, exist_ok=True)
    
    print(f"Starting download for {model_id}...")
    print(f"Files will be saved strictly to: {local_path}")
    
    # Download the repository directly to your specified folder
    snapshot_download(
        repo_id=model_id,
        local_dir=local_path,
        local_dir_use_symlinks=False,  # CRITICAL for HPC clusters
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"]  # Skip legacy weight formats
    )
    
    print("Download complete. Model is ready for offline HPC nodes.")

if __name__ == "__main__":
    download_qwen_offline()