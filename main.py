import os
import glob
import json
from pathlib import Path
from PIL import Image
import imagehash
import faiss
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple, List, Set
import torch
import torch.nn.functional as F
from lightning_cloud.utils.data_connection import add_s3_connection
from utils.laion_streaming_dataset import LAOINStreamingDataset
from utils.HF_dataset_eval import HFDataset_eval
from utils.HF_dataset import HFDataset
from utils.optimize_hf_to_lightning import optimize_hf_to_lightning
import clip
import textwrap
from openai import OpenAI
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============= Parameter Configuration =============
dataset_name = "pets"  # Options: "food101", "cars", "country211", "aircraft", "pets", "cub"

threshold = 4  # Maximum Hamming distance for considering images as duplicates (inclusive)
k = 5  # Number of duplicate matches to show per image

batch_size = 64  # Batch size for processing images
num_workers = 4  # Number of worker processes for data loading
sim_threshold = 0.88  # CLIP similarity threshold for filtering duplicates

api_key = None # enter personal OpenAI API Key to enable caption analysis
# ================================================

# Load classes
try:
    classes = json.load(open(f"data/classes/classes_{dataset_name}.json", "r"))
except FileNotFoundError:
    classes = None
    print("No classes file found, using raw labels")

# Load the target dataset from Huggingface
if dataset_name == "food101":
    hf_dataset = load_dataset("clip-benchmark/wds_food101")
elif dataset_name == "cars":
    hf_dataset = load_dataset("clip-benchmark/wds_cars")
elif dataset_name == "country211":
    hf_dataset = load_dataset("clip-benchmark/wds_country211")
elif dataset_name == "fgvc-aircraft":
    hf_dataset = load_dataset("clip-benchmark/wds_fgvc_aircraft")
elif dataset_name == "pets":
    hf_dataset = load_dataset("clip-benchmark/wds_vtab-pets")
elif dataset_name == "cub":
    hf_dataset = load_dataset("lxs784/cub-200-2011-clip-benchmark")
else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")

# Determine image key using the first available split
first_split = next(iter(hf_dataset.keys()))
if "webp" in hf_dataset[first_split][0] and hf_dataset[first_split][0]["webp"] is not None:
    image_key = "webp"
elif hf_dataset[first_split][0]["jpg"] is not None:
    image_key = "jpg"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

client = OpenAI(api_key=api_key) if api_key else None

# Load LAION-400M dataset
add_s3_connection("laoin-400m")
laion = LAOINStreamingDataset(input_dir="/teamspace/s3_connections/laoin-400m")

def hex_to_vector(hex_str, vector_dim=16):
    """
    Convert a 16-character hex string to a 64-bit binary vector. 
    Each hex digit is converted to a 4-bit binary number.
    Ensure the hex string is exactly 16 characters long and binary vector is exactly 64 bits long.
    """
    if hex_str is None:
        return [0] * vector_dim * 4

    if len(hex_str) != vector_dim:
        raise ValueError(f"Hex string length ({len(hex_str)}) does not match expected dimension ({vector_dim}).")
    
    vector = []
    for digit in hex_str:
        if digit not in "0123456789abcdef":
            raise ValueError("Invalid hex string")

        binary_str = bin(int(digit, 16))[2:].zfill(4)
        vector.extend([int(bit) for bit in binary_str])

    if len(vector) != vector_dim * 4:
        raise ValueError("Hex string did not convert to the expected number of bits")
    return vector

def find_duplicates(dataset_name: str, dataloader, threshold: int, binary_index_phash, k: int = 5) -> dict:
    """
    Find duplicate images between target dataset and LAION-400M.
    Parameters:
        dataset_name (str): The name of the dataset to process.
        dataloader (iterable): An iterable (e.g., a DataLoader) that yields batches of data.
            Each batch is expected to be a tuple containing:
                - an ignored element (e.g., image data),
                - texts (list or similar),
                - ahashes (list or similar),
                - phashes (list of hexadecimal pHash strings),
                - uids (list of unique identifiers for each image).
        threshold (int): The maximum Hamming distance to consider two images as duplicates.
        binary_index_phash: An object with a method `range_search` that takes a packed query array and threshold,
            and returns search results (lims, D_range, I_range) for duplicate detection.
        hex_to_vector (callable): A function that converts a hexadecimal pHash string into a vector of integers.
        k (int, optional): Maximum number of duplicate matches to retain per image. Defaults to 5.

    Returns:
        dict: A dictionary mapping image unique IDs (uids) to a list of duplicate indices found.
              Also saves intermediate and combined results in JSON files under the designated directory.
    """
    results = {}
    part = 0
    json_dir = f"/teamspace/studios/this_studio/data/intermediate/{dataset_name}/match_indices_{threshold}"
    os.makedirs(json_dir, exist_ok=True)

    for i, (_, texts, ahashes, phashes, uids) in enumerate(tqdm(dataloader, desc=f"Finding duplicates in {dataset_name}")):
        query_vectors = np.array([hex_to_vector(x, 16) for x in phashes], dtype='uint8')
        queries_packed = np.packbits(query_vectors, axis=1).reshape(len(phashes), 8)

        lims, D_range, I_range = binary_index_phash.range_search(queries_packed, threshold)

        for q in range(queries_packed.shape[0]):
            start = lims[q]
            end = lims[q + 1]
            if start == end:
                continue
            match_indices = I_range[start:end].tolist()
            if len(match_indices) > 0:
                results[uids[q]] = match_indices
            if len(results) == 100:
                with open(os.path.join(json_dir, f"results_{part}.json"), "w") as f:
                    json.dump(results, f)
                    tqdm.write(f"part {part} saved!")
                results = {} # reset
                part += 1
                
    if len(results) > 0:
        with open(os.path.join(json_dir, f"results_{part}.json"), "w") as f:
            json.dump(results, f)

    # put all results in one json file
    json_files = glob.glob(os.path.join(json_dir, "*.json"))

    results = {}
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
        results.update(data)

    with open(os.path.join(json_dir, "combined_results.json"), "w") as f:
        json.dump(results, f)
    print(f"Combined results saved to {os.path.join(json_dir, 'combined_results.json')}, total duplicate images: {len(results)}")
    return results

def resize_image(image, target_size=(256, 256)):
    return image.resize(target_size, Image.Resampling.LANCZOS)

def filter_and_visualize_duplicates(dataset_name: str, dataset: HFDataset, results: dict, laion, k: int = k, sim_threshold: float = 0.88):
    """
    Visualize duplicate pairs of images between target dataset and LAION-400M.
    
    Parameters:
        target_dataset: The dataset to find duplicates for
        laion_dataset: The LAION-400M dataset
        duplicates: List of (target_uid, laion_uid, distance) tuples
        save_dir: Directory to save visualization images
    """
    output_dir = f"data/intermediate/{dataset_name}/plots"
    correct_dir = os.path.join(output_dir, "correct")
    incorrect_dir = os.path.join(output_dir, "incorrect")

    output_indices = f"data/final/{dataset_name}/final_results.json"

    final_results = {}
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)
    cols = k + 2
    for uid, match_indices in tqdm(results.items(), desc=f"plotting duplicate images for {dataset_name}"):
        fig, axes = plt.subplots(1, cols, figsize=(cols * 3, 3))
        axes[0].text(0.5, 0.5, uid, fontsize=24, ha='center', va='center')
        axes[0].axis("off")

        original_image, original_text, ahash, phash= dataset.get_by_id(uid)
        original_image_resized = resize_image(original_image)
        axes[1].imshow(original_image_resized)
        wrapped_caption = "\n".join(textwrap.wrap(original_text, width=24))
        axes[1].set_title(wrapped_caption)
        axes[1].axis('off')
        orig_input = preprocess(original_image).unsqueeze(0).to(device)
        with torch.no_grad():
            orig_features = model.encode_image(orig_input)
            orig_features /= orig_features.norm(dim=-1, keepdim=True)

        correct = 0
        for j in range (k):
            ax = axes[j + 2]
            if j >= len(match_indices):
                ax.imshow(np.ones((1, 1, 3)))
            else:
                idx = match_indices[j]
                match_image, match_text, _ = laion[idx]
                match_input = preprocess(match_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    match_features = model.encode_image(match_input)
                    match_features /= match_features.norm(dim=-1, keepdim=True)
                similarity = (orig_features @ match_features.T).item()
                
                laion_phash = imagehash.phash(match_image)
                p_dist = abs(phash - laion_phash)
                ax.imshow(match_image)
                caption_match = "dist: " + str(p_dist) + " " + match_text
                wrapped_lines = textwrap.wrap(caption_match, width=24)
                wrapped_caption_match = "\n".join(wrapped_lines[:2])
                ax.set_title(wrapped_caption_match, fontsize=8)
                if similarity >= sim_threshold:
                    correct += 1
                    if uid not in final_results:
                        final_results[uid] = [idx]
                    else:
                        final_results[uid].append(idx)
            ax.axis('off')
        plt.tight_layout()
        if correct > 0:
            plt.savefig(os.path.join(correct_dir, f"{uid}.png"))
        else:
            plt.savefig(os.path.join(incorrect_dir, f"{uid}.png")) # save to another directory
        plt.close(fig)

    with open(output_indices, "w") as f:
        json.dump(final_results, f)
    print(f"Filtered Visualizations saved to {output_dir} and {output_indices}")

def classify_caption_gpt(caption, class_name):
    prompt = f"""
        You are a classification system that determines if a caption is relevant to a given class name.
        Instructions:
        1. Identify key words and meaningful components in both the class name and the caption. For compound class names (e.g., "Carolina wren" or "Golden retriever"), consider both the complete name and its individual components. A match on any significant part should be considered as evidence of relevance.
        2. Expand the meaning of the class name by including synonyms, related terms, hypernyms, hyponyms, and inferred concepts that capture the broader or more specific context of the class. For instance, if the class name implies a specific category, also consider general or related terms that fall under the same category.
        3. Matching Criteria:
        - If the caption contains the full class name, any significant component of it, or any of the expanded related terms, then return "1" (indicating relevance).
        - If the caption shows no connection to the class name or any of its expanded semantic associations, return "2" (indicating no relation).


        Class Name: {class_name}
        Caption: {caption}

        Return only "1" or "2". No explanations.
        """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20
    )
    cleaned_response = response.choices[0].message.content.strip().lower().replace('"', '').replace("'", "")
    print(f"class: {class_name}, captions: {caption}, result: {cleaned_response}")
    return cleaned_response

def is_fully_included(class_name: str, caption: str) -> bool:
    """
    Check if all words from class_name are fully included in the caption.
    
    Args:
        class_name: The class name to check
        caption: The caption to check against
        
    Returns:
        True if all words are included, False otherwise
    """
    # Normalize caption by replacing hyphens and underscores with spaces.
    normalized_caption = re.sub(r'[-_]', ' ', caption)
    
    # Split the class name into words.
    words = re.split(r'[\W_]+', class_name)
    words = [word for word in words if word]

    # For each word, search in the normalized caption. Allow optional trailing digits.
    return all(
        re.search(r'\b' + re.escape(word) + r'\d*\b', normalized_caption, flags=re.IGNORECASE)
        for word in words
    )

def process_item(item: Tuple[str, List[int]], dataset: HFDataset) -> Tuple[str, int, int, int]:
    """
    Process a single item to classify its captions.
    
    Args:
        item: Tuple of (uid, indices)
        dataset: The target dataset
        laion: The LAION dataset
        client: Optional OpenAI client for GPT classification
        
    Returns:
        Tuple of (uid, correct, relevant, irrelevant)
    """
    uid, indices = item
    _, class_name, _, _ = dataset.get_by_id(uid)

    correct = 0
    relevant = 0
    irrelevant = 0

    for index in indices:
        laion_caption = laion[index][1]
        if is_fully_included(class_name, laion_caption):
            correct = 1
        elif client is not None:
            response = classify_caption_gpt(laion_caption, class_name)
            if response == "1":
                relevant = 1
            elif response == "2":
                irrelevant = 1
            else:
                print(f"ERROR: unexpected response for {uid}, {index}: {response}")

    return uid, correct, relevant, irrelevant

def categorize_results(all_captions: List[str], correct_captions: List[str], 
                      relevant_captions: List[str], irrelevant_captions: List[str]) -> Dict[str, List[str]]:
    """
    Categorize results into different sets based on caption classifications.
    
    Args:
        all_captions: List of all caption UIDs
        correct_captions: List of UIDs with correct captions
        relevant_captions: List of UIDs with relevant captions
        irrelevant_captions: List of UIDs with irrelevant captions
        
    Returns:
        Dictionary mapping category names to lists of UIDs
    """
    categories = {
        "only_correct": [],
        "only_relevant": [],
        "only_irrelevant": [],
        "correct_and_relevant": [],
        "correct_and_irrelevant": [],
        "relevant_and_irrelevant": [],
        "mixed": []
    }
    
    for uid in all_captions:
        is_correct = uid in correct_captions
        is_relevant = uid in relevant_captions
        is_irrelevant = uid in irrelevant_captions
        
        if is_correct and not is_relevant and not is_irrelevant:
            categories["only_correct"].append(uid)
        elif is_relevant and not is_correct and not is_irrelevant:
            categories["only_relevant"].append(uid)
        elif is_irrelevant and not is_correct and not is_relevant:
            categories["only_irrelevant"].append(uid)
        elif is_correct and is_relevant and not is_irrelevant:
            categories["correct_and_relevant"].append(uid)
        elif is_correct and is_irrelevant and not is_relevant:
            categories["correct_and_irrelevant"].append(uid)
        elif is_relevant and is_irrelevant and not is_correct:
            categories["relevant_and_irrelevant"].append(uid)
        elif is_correct and is_relevant and is_irrelevant:
            categories["mixed"].append(uid)
        else:
            print(f"Error! Unexpected classification for uid: {uid}")
    
    return categories

def process_item_wrapper(item_dataset_tuple):
    """
    A wrapper function to call process_item with the provided arguments.
    This avoids using a lambda (which can cause pickling issues).
    """
    item, dataset = item_dataset_tuple
    return process_item(item, dataset)


def analyze_duplicates(dataset_name: str, dataset: HFDataset, 
                       final_results: Dict[str, List[int]]) -> None:
    """
    Analyze duplicates and classify their captions.
    
    Args:
        dataset_name: Name of the target dataset.
        dataset: The target dataset.
        final_results: Dictionary mapping UIDs to lists of duplicate indices.
    """
    print(f"Processing results of {dataset_name}...")
    
    # Initialize lists for different caption types
    all_captions = []
    correct_captions = []
    relevant_captions = []
    irrelevant_captions = []
    
    # Prepare items for processing
    items = list(final_results.items())
    
    # Use ProcessPoolExecutor and as_completed to update the progress bar as tasks finish.
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Prepare tuples so that both 'item' and 'dataset' can be passed.
        futures = {executor.submit(process_item_wrapper, (item, dataset)): item for item in items}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing duplicates"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing item: {e}")
    
    # Collect results from processed items
    for uid, correct, relevant, irrelevant in results:
        all_captions.append(uid)
        correct_captions.extend([uid] * correct)
        relevant_captions.extend([uid] * relevant)
        irrelevant_captions.extend([uid] * irrelevant)
    
    # Save initial results to JSON files
    output_dir = f"data/final/{dataset_name}/duplicate_categories"
    os.makedirs(output_dir, exist_ok=True)
    json.dump(all_captions, open(os.path.join(output_dir, "all_captions.json"), "w"))
    json.dump(correct_captions, open(os.path.join(output_dir, "correct_captions.json"), "w"))
    json.dump(relevant_captions, open(os.path.join(output_dir, "relevant_captions.json"), "w"))
    json.dump(irrelevant_captions, open(os.path.join(output_dir, "irrelevant_captions.json"), "w"))
    
    # Categorize and save final results
    categories = categorize_results(all_captions, correct_captions, relevant_captions, irrelevant_captions)
    for category, uids in categories.items():
        json.dump(uids, open(os.path.join(output_dir, f"{category}.json"), "w"))
        print(f"{category}: {len(uids)}")
    
    print("Analysis complete!")

def process_split(split_name: str, hf_split_dataset):
    """
    Process a single split of the dataset.
    
    Parameters:
        split_name: Name of the split (e.g., "train", "test")
        hf_split_dataset: The HuggingFace dataset split
    """
    print(f"\nProcessing split: {split_name}")
    target_dataset_name = f"{dataset_name}-{split_name}"
    
    # Process target dataset
    target_optimized_dir = f"data/optimized_dataset/{target_dataset_name}"
    if not os.path.exists(os.path.join(target_optimized_dir, "index.json")):
        optimize_hf_to_lightning(hf_split_dataset, target_optimized_dir, image_key=image_key)
    
    # Create datasets
    target_dataset = HFDataset(
        index_file="index.json",
        root_dir=target_optimized_dir,
        lookup=classes if classes else None,
    )
    dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # load faiss index
    binary_index_phash = faiss.read_index_binary("data/lightning_binary_index.bin")
    
    # Find duplicates
    existing_duplicates = f"data/intermediate/{target_dataset_name}/match_indices_{threshold}/combined_results.json"
    if os.path.exists(existing_duplicates):
        duplicates = json.load(open(existing_duplicates, "r"))
    else:
        duplicates = find_duplicates(target_dataset_name, dataloader, threshold, binary_index_phash, k)

    # Visualize duplicates
    filter_and_visualize_duplicates(target_dataset_name, target_dataset, duplicates, laion)

    # After finding duplicates, analyze the captions
    if api_key:
        final_results_file = f"data/final/{target_dataset_name}/final_results.json"
        final_results = json.load(open(final_results_file, "r"))
        analyze_duplicates(target_dataset_name, target_dataset, final_results)

def main():
    # Get all available splits
    splits = hf_dataset.keys()
    print(f"Available splits: {splits}")
    
    # Process each split
    for split in splits:
        process_split(split, hf_dataset[split])

if __name__ == "__main__":
    main() 