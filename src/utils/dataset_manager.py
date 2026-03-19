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
        Includes support for extracting .tar.gz archives from OpenXLab structure.
        """
        if scenes is None:
            scenes = self.SCENES
            
        os.makedirs(self.root_dir, exist_ok=True)
        
        # OpenXLab often creates a subfolder based on the repo name
        repo_subfolder = self.REPO.replace("/", "___") # OpenDataLab___Mill_19
        search_paths = [
            os.path.join(self.root_dir, "Mill_19.tar.gz"),
            os.path.join(self.root_dir, "raw", "Mill_19.tar.gz"),
            os.path.join(self.root_dir, repo_subfolder, "raw", "Mill_19.tar.gz"),
            os.path.join(self.root_dir, repo_subfolder, "Mill_19.tar.gz")
        ]
        
        target_tar = next((p for p in search_paths if os.path.exists(p)), None)
            
        if target_tar:
            print(f"Found archive at {target_tar}. Attempting extraction...")
            return self.extract_archive(target_tar)

        if not self.login():
            print("Failed to login. Please ensure AK/SK are set.")
            return False
            
        print(f"Requesting download of {self.REPO}...")
        try:
            # Re-check paths after download attempt (in case of partial/background completion)
            get(dataset_repo=self.REPO, target_path=self.root_dir)
            print(f"Download call returned.")
            
            # Search again
            target_tar = next((p for p in search_paths if os.path.exists(p)), None)
            if target_tar:
                return self.extract_archive(target_tar)
        except Exception as e:
            print(f"Download interaction log: {e}")
            target_tar = next((p for p in search_paths if os.path.exists(p)), None)
            if target_tar:
                print("File is present despite error/timeout. Extracting...")
                return self.extract_archive(target_tar)
            return False
        return True

    def extract_to(self, archive_path, dest_dir):
        """Extracts a specific archive to a destination directory."""
        import tarfile
        try:
            with tarfile.open(archive_path, 'r:*') as tar:
                tar.extractall(path=dest_dir)
            print(f"Extracted {os.path.basename(archive_path)} to {dest_dir}.")
            return True
        except Exception as e:
            print(f"Failed to extract {archive_path}: {e}")
            return False

    def extract_archive(self, path):
        """Main archive extraction (recursive by default)."""
        import tarfile
        print(f"Extracting {path} to {self.root_dir}...")
        try:
            with tarfile.open(path, 'r:*') as tar:
                tar.extractall(path=self.root_dir)
            print(f"Extraction of {os.path.basename(path)} complete.")
            # Normalization and nested extraction
            self._normalize_folders()
            return True
        except Exception as e:
            print(f"Extraction failed for {path}: {e}")
            return False

    def _normalize_folders(self):
        """
        Processes the root_dir to find and extract nested scene archives.
        """
        print(f"Normalizing folders in {self.root_dir}...")
        
        # Check for nested scene archives like 'rubble-pixsfm.tgz'
        for root, dirs, files in os.walk(self.root_dir):
            for f in files:
                if f.endswith(('.tgz', '.tar.gz')) and f != "Mill_19.tar.gz":
                    archive_path = os.path.join(root, f)
                    scene_name = f.split("-")[0].lower()
                    if scene_name in self.SCENES:
                        scene_dest = os.path.join(self.root_dir, scene_name)
                        # Avoid repeated extraction if data already exists
                        if not os.path.exists(os.path.join(scene_dest, "train", "rgbs")):
                            print(f"Found nested archive for {scene_name}: {archive_path}.")
                            os.makedirs(scene_dest, exist_ok=True)
                            self.extract_to(archive_path, scene_dest)

        # Final check/move for already extracted folders starting with scene name
        for scene in self.SCENES:
            for root, dirs, files in os.walk(self.root_dir):
                if root != self.root_dir: continue # Only look at top level or immediate children
                for d in dirs:
                    if d.startswith(scene) and d != scene:
                        src = os.path.join(root, d)
                        dst = os.path.join(self.root_dir, scene)
                        print(f"Normalizing scene folder: {src} -> {dst}")
                        if os.path.exists(dst): shutil.rmtree(dst)
                        shutil.move(src, dst)

    def preprocess(self):
        """
        Final check and metadata generation.
        """
        print("Preprocessing Mill 19 dataset...")
        self._normalize_folders()
        
        for scene in self.SCENES:
            scene_dir = os.path.join(self.root_dir, scene)
            if not os.path.exists(scene_dir):
                continue
                
            # Generate metadata if images are found
            img_dir = None
            possible_dirs = [
                os.path.join(scene_dir, "images"),
                os.path.join(scene_dir, "train", "rgbs"),
                os.path.join(scene_dir, "rgbs")
            ]
            for d in possible_dirs:
                if os.path.exists(d):
                    img_dir = d
                    break
            
            if img_dir:
                metadata_file = os.path.join(scene_dir, "metadata.json")
                if not os.path.exists(metadata_file):
                    imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                    if imgs:
                        print(f"Generating metadata for {scene}...")
                        meta = {
                            "scene": scene,
                            "num_images": len(imgs),
                            "image_path": os.path.relpath(img_dir, scene_dir),
                            "intrinsic": [500.0, 500.0, self._get_img_size(os.path.join(img_dir, imgs[0]))]
                        }
                        with open(metadata_file, 'w') as f:
                            json.dump(meta, f, indent=4)
                        
        print("Preprocessing complete.")

    def _get_img_size(self, path):
        from PIL import Image
        with Image.open(path) as img:
            return img.size # (W, H)

if __name__ == "__main__":
    manager = Mill19DatasetManager()
    if manager.download_dataset():
        manager.preprocess()
