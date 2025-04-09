import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def optimize_hf_to_lightning(hf_dataset, output_dir: str, image_key: str = "webp", id_key: str = "__key__", label_key: str = "cls", is_streaming: bool = False) -> str:
    """
    Iterates over the Hugging Face dataset and saves each sample to disk in a format
    that Lightning's StreamingDataset can read. An index file (index.json) is created.
    
    Parameters:
      hf_dataset: The Hugging Face dataset (can be streaming or in-memory)
      output_dir: Directory where the optimized dataset will be stored.
      image_key: Field name in the dataset containing image data.
      id_key: Field name to use as a unique identifier.
      label_key: Field name containing label or class information.
      is_streaming: Whether the dataset is streaming or not
    Returns:
      The output directory path (which contains the data and index).
    """
    os.makedirs(output_dir, exist_ok=True)
    index = {}
    
    # Iterate over the dataset and write each sample.
    for sample in tqdm(hf_dataset):
        uid = sample[id_key]
        # Define a file path for the image.
        image_filename = f"{uid}.png"
        image_path = os.path.join(output_dir, image_filename)
        
        # Get the image. Depending on your dataset, it might already be a PIL Image.
        image = sample[image_key]
        if not isinstance(image, Image.Image):
            # If image is not a PIL image, try converting it.
            image = Image.fromarray(image)
            
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Save the image in PNG format.
        image.save(image_path, format="PNG")
        
        # Record metadata in the index.
        index[uid] = {
            "image_path": image_filename,  # Store relative path
            "label": sample[label_key],
        }
    
    # Write out the index file.
    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f)
    
    return output_dir