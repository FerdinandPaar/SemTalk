from huggingface_hub import snapshot_download
import os

# Create directories
os.makedirs('facebook/hubert-large-ls960-ft', exist_ok=True)
os.makedirs('Systran/faster-whisper-large-v3', exist_ok=True)

print("Downloading HuBERT (1.3GB)...")
snapshot_download(
    repo_id="facebook/hubert-large-ls960-ft", 
    local_dir="facebook/hubert-large-ls960-ft",
    resume_download=True
)

print("Downloading Whisper (3GB)...")
snapshot_download(
    repo_id="Systran/faster-whisper-large-v3", 
    local_dir="Systran/faster-whisper-large-v3",
    resume_download=True
)

print("✓ All downloads complete!")
