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

    def extract_archive(self, path):
        import tarfile
        print(f"Extracting {path} to {self.root_dir}... (This may take a while)")
        try:
            with tarfile.open(path, 'r:gz') as tar:
                tar.extractall(path=self.root_dir)
            print("Extraction complete.")
            # Normalization: Move extracted folders to root_dir if they are nested
            self._normalize_folders()
            return True
        except Exception as e:
            print(f"Extraction of {os.path.basename(path)} complete.")
            if recursive:
                self._normalize_folders()
            return True
        except Exception as e:
            print(f"Extraction failed for {path}: {e}")
            return False

    def _normalize_folders(self):
        """
        Moves scene folders from subdirectories to the root data directory.
        Now also handles nested .tgz files (e.g., rubble-pixsfm.tgz).
        """
        print(f"Normalizing folders in {self.root_dir}...")
        
        # First, check for any nested archives that need extraction
        for root, dirs, files in os.walk(self.root_dir):
            for f in files:
                if f.endswith(('.tgz', '.tar.gz')) and f != "Mill_19.tar.gz":
                    archive_path = os.path.join(root, f)
                    print(f"Found nested archive: {archive_path}. Extracting...")
                    self.extract_archive(archive_path, recursive=False)

        found_any = False
        for scene in self.SCENES:
            # Search for scene folder in all extracted subfolders
            # Logic: match if folder name starts with scene name (to catch 'rubble-pixsfm' as 'rubble')
            for root, dirs, files in os.walk(self.root_dir):
                for d in dirs:
                    if d.startswith(scene):
                        src = os.path.join(root, d)
                        dst = os.path.join(self.root_dir, scene)
                        if src != dst:
                            print(f"Found match for {scene} at {src}. Moving to {dst}...")
                            try:
                                if os.path.exists(dst):
                                    shutil.rmtree(dst)
                                shutil.move(src, dst)
                                print(f"Successfully moved {scene}.")
                                found_any = True
                            except Exception as e:
                                print(f"Error moving {scene}: {e}")
                        else:
                            # Already at destination, but might be named 'rubble-pixsfm'
                            if d != scene:
                                target = os.path.join(self.root_dir, scene)
                                os.rename(src, target)
                            found_any = True
                        break # Found this scene
                if found_any: break
                    
        if not found_any:
            print("Warning: No scene folders found. Listing all directory content for debugging:")
            for root, dirs, files in os.walk(self.root_dir):
                print(f"  Dir: {root} -> Subdirs: {dirs} -> Files: {files[:5]}")

    def preprocess(self):
        """
        Organizes the downloaded data into a consistent format for SERAPH.
        """
        print("Preprocessing Mill 19 dataset...")
        self._normalize_folders()
        
        for scene in self.SCENES:
            scene_dir = os.path.join(self.root_dir, scene)
            if not os.path.exists(scene_dir):
                continue
                
            # Check for images folder (Mill 19 often has it inside another subfolder)
            img_target = os.path.join(scene_dir, "images")
            if not os.path.exists(img_target):
                for root, dirs, files in os.walk(scene_dir):
                    if "images" in dirs:
                        src = os.path.join(root, "images")
                        print(f"Found images folder at {src}. Moving to {img_target}")
                        if os.path.exists(img_target): shutil.rmtree(img_target)
                        shutil.move(src, img_target)
                        break
            
            metadata_file = os.path.join(scene_dir, "metadata.json")
            if not os.path.exists(metadata_file):
                if os.path.exists(img_target):
                    imgs = [f for f in os.listdir(img_target) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                    if imgs:
                        print(f"Generating metadata for {scene}...")
                        meta = {
                            "scene": scene,
                            "num_images": len(imgs),
                            "image_path": "images",
                            "intrinsic": [500.0, 500.0, self._get_img_size(os.path.join(img_dir, imgs[0]))]
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
