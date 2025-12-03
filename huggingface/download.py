# pip install huggingface-hub

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from huggingface_hub import snapshot_download

repo_id = "BAAI/bge-large-en-v1.5"

folder_path = snapshot_download(
    repo_id=repo_id,
    repo_type="model",  # "model" or "dataset"
    local_dir=repo_id.split("/")[-1],
)
print("folder_path:", folder_path)