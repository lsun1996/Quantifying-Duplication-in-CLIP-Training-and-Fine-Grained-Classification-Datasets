import torch
import os
import json
from PIL import Image
import imagehash
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class HFDataset_eval(Dataset):

    def __init__(self, root_dir, index_file, lookup=None, transform=None):
        self.root_dir = root_dir
        with open(os.path.join(root_dir, index_file), "r") as f:
            self.index_data = json.load(f)
        self.lookup = lookup
        self.samples = list(self.index_data.items())
        self.uid_to_sample = dict(self.samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        uid, sample = self.samples[index]
        image_path = os.path.join(self.root_dir, sample["image_path"])
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        text = self.lookup[sample["label"]] if self.lookup else sample["label"]

        return image_bytes, text, uid

    def get_by_id(self, uid):
        """
        Retrieve a raw PIL image and metadata by its unique identifier.
        """
        if uid not in self.uid_to_sample:
            raise KeyError(f"UID {uid} not found in dataset.")
        sample = self.uid_to_sample[uid]
        image_path = os.path.join(self.root_dir, sample["image_path"])
        pil_image = Image.open(image_path).convert("RGB")
        text = self.lookup[sample["label"]] if self.lookup else sample["label"]

        return pil_image, text