import os
import subprocess
import shutil
import json
import torch
from tqdm import tqdm

try:
    import openxlab
    from openxlab.dataset import info, get, download
except ImportError:
    openxlab = None

class Mill19DatasetManager:
    """
    Automates the Mill 19 dataset lifecycle:
    - Login to OpenDataLab via OpenXLab
    - Download specific scenes (rubble, residential, sci-art, building)
    - Preprocess and organize for SERAPH training
    """
    
    REPO = "OpenDataLab/Mill_19"
    SCENES = ["rubble", "residential", "sci-art", "building"]
    
    def __init__(self, root_dir="data/mill19", ak=None, sk=None):
        self.root_dir = root_dir
        # Provided credentials for automated download
        self.ak = ak or "e8z4v95bjkmgngzxrj2e"
        self.sk = sk or "dm6wblqp7xg0kav3pnov6zjz5m8z5zay19roqepb"
        
    def login(self):
        if openxlab is None:
            print("Error: openxlab package not installed. Run 'pip install openxlab'.")
            return False
            
        if not self.ak or not self.sk:
            print("Warning: Access Key (AK) or Secret Key (SK) missing. Login may fail.")
            return False
            
        try:
            openxlab.login(ak=self.ak, sk=self.sk)
            print("Logged into OpenXLab successfully.")
            return True
        except Exception as e:
            print(f"Login failed: {e}")
            return False

    def download_dataset(self, scenes=None):
        """
        Downloads requested scenes from Mill 19 repo.
        """
        if scenes is None:
            scenes = self.SCENES
            
        os.makedirs(self.root_dir, exist_ok=True)
        
        if not self.login():
            print("Failed to login. Please ensure AK/SK are set.")
            return False
            
        for scene in scenes:
            scene_path = os.path.join(self.root_dir, scene)
            if os.path.exists(scene_path):
                print(f"Scene {scene} already exists at {scene_path}. Skipping download.")
                continue
                
            print(f"Downloading scene: {scene}...")
            try:
                # get() downloads the whole repo or a sub-path depends on implementation
                # The screenshot shows: openxlab dataset get --dataset-repo OpenDataLab/Mill_19
                # We target the specific source path inside the repo if possible
                get(dataset_repo=self.REPO, target_path=self.root_dir)
                print(f"Successfully downloaded {scene}.")
            except Exception as e:
                print(f"Failed to download {scene}: {e}")
                return False
        return True

    def preprocess(self):
        """
        Organizes the downloaded data into a consistent format for SERAPH.
        Format: data/mill19/<scene>/[images, colmap_poses, metadata.json]
        """
        print("Preprocessing Mill 19 dataset...")
        for scene in self.SCENES:
            scene_dir = os.path.join(self.root_dir, scene)
            if not os.path.exists(scene_dir):
                continue
                
            # Example organization logic:
            # 1. Look for images/ directory
            # 2. Extract metadata if in a non-standard format
            # 3. Create a unified dataset.json for the dataloader
            
            metadata_file = os.path.join(scene_dir, "metadata.json")
            if not os.path.exists(metadata_file):
                # Create a simple metadata file if missing
                img_dir = os.path.join(scene_dir, "images")
                if os.path.exists(img_dir):
                    imgs = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
                    meta = {
                        "scene": scene,
                        "num_images": len(imgs),
                        "image_path": "images",
                        "intrinsic": [500.0, 500.0, self._get_img_size(os.path.join(img_dir, imgs[0]))] if imgs else None
                    }
                    with open(metadata_file, 'w') as f:
                        json.dump(meta, f, indent=4)
                        
        print("Preprocessing complete.")

    def _get_img_size(self, path):
        # Quick helper to get dimensions
        from PIL import Image
        with Image.open(path) as img:
            return img.size # (W, H)

if __name__ == "__main__":
    # Test manager
    manager = Mill19DatasetManager()
    if manager.download_dataset():
        manager.preprocess()
