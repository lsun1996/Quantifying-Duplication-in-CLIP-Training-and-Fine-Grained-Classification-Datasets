import os
import json
from PIL import Image
import imagehash
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, Tuple, List

class HFDataset(Dataset):
    def __init__(self, root_dir: str, index_file: str, lookup: Optional[List[str]] = None):
        self.root_dir = root_dir
        with open(os.path.join(root_dir, index_file), "r") as f:
            self.index_data = json.load(f)
        self.lookup = lookup
        self.samples = list(self.index_data.items())
        self.uid_to_sample = dict(self.samples)

    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, index: int) -> Tuple[int, str, str, str, str]:
        uid, sample = self.samples[index]
        image_path = os.path.join(self.root_dir, sample["image_path"])
        pil_image = Image.open(image_path).convert("RGB")
        text = self.lookup[sample["label"]] if self.lookup else sample["label"]

        ahash = str(imagehash.average_hash(pil_image))
        phash = str(imagehash.phash(pil_image))

        return index, text, ahash, phash, uid

    def get_by_id(self, uid: str) -> Tuple[Image.Image, str, imagehash.ImageHash, imagehash.ImageHash]:
        """
        Retrieve a raw PIL image and metadata by its unique identifier.
        """
        sample = self.uid_to_sample[uid]
        image_path = os.path.join(self.root_dir, sample["image_path"])
        pil_image = Image.open(image_path).convert("RGB")
        text = self.lookup[sample["label"]] if self.lookup else sample["label"]
        ahash = imagehash.average_hash(pil_image)
        phash = imagehash.phash(pil_image)

        return pil_image, text, ahash, phash