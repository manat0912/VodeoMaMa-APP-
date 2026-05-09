#!/usr/bin/env python
"""
Cross-platform checkpoint download script for VideoMaMa demo
"""
import os
import sys
from pathlib import Path
import shutil
import urllib.request

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import snapshot_download

class CheckpointDownloader:
    def __init__(self):
        self.checkpoint_dir = Path(__file__).parent / 'checkpoints'
        
    def download_file(self, url, output_path):
        if output_path.exists():
            print(f"✓ Exists: {output_path}")
            return True
            
        print(f"Downloading {url} to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url) as response, open(output_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            print("✓ Download successful")
            return True
        except Exception as e:
            print(f"❌ Failed: {e}")
            if output_path.exists():
                output_path.unlink()
            return False

    def run(self):
        print(f"Checkpoints directory: {self.checkpoint_dir}")

        # 1. SAM 2.1
        sam2_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        sam2_path = self.checkpoint_dir / "sam2" / "sam2.1_hiera_large.pt"
        self.download_file(sam2_url, sam2_path)

        # 2. SVD
        print("\nChecking/Downloading SVD model...")
        svd_path = self.checkpoint_dir / "stable-video-diffusion-img2vid-xt"
        try:
            snapshot_download(repo_id="stabilityai/stable-video-diffusion-img2vid-xt", 
                            local_dir=svd_path, 
                            local_dir_use_symlinks=False)
            print("✓ SVD Ready")
        except Exception as e:
            print(f"❌ SVD Download failed: {e}")

        # 3. VideoMaMa
        print("\nChecking/Downloading VideoMaMa model...")
        videomama_path = self.checkpoint_dir / "videomama"
        try:
            snapshot_download(repo_id="SammyLim/VideoMaMa", 
                            allow_patterns=["unet/*"],
                            local_dir=videomama_path, 
                            local_dir_use_symlinks=False)
            print("✓ VideoMaMa Ready")
        except Exception as e:
            print(f"❌ VideoMaMa Download failed: {e}")

if __name__ == "__main__":
    CheckpointDownloader().run()
