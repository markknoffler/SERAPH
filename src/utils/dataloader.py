import torch
from torch.utils.data import Dataset
import os
import json
from PIL import Image
import numpy as np

class Mill19Dataset(Dataset):
    """
    Dataset for Mill 19 (Rubble, Residential, Building, Sci-Art).
    Loads images and precomputed masks/features from EDN if available.
    """
    def __init__(self, root_dir, scene="rubble", split="train", transform=None):
        self.root_dir = os.path.join(root_dir, scene)
        self.split = split
        self.transform = transform
        
        # In Mill 19 / Mega-NeRF format, images are often in a subfolder
        # We look for metadata created by DatasetManager
        meta_path = os.path.join(self.root_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.meta = json.load(f)
        else:
            self.meta = {"image_path": "images"}
            
        self.img_dir = os.path.join(self.root_dir, self.meta.get("image_path", "images"))
        
        if os.path.exists(self.img_dir):
            self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))])
        else:
            self.image_files = []
            print(f"Warning: Image directory {self.img_dir} not found.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        else:
            # Default to tensor and normalize
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            
        # Return image and some metadata if needed
        return img
