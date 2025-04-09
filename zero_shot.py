from datasets import load_dataset, concatenate_datasets, DatasetDict
import json
import numpy as np
import torch
import open_clip
from open_clip import tokenizer
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os

# ============= Parameter Configuration =============
dataset_name = "cub"
batch_size = 64
num_workers = 4
# ================================================

train_available = True
if dataset_name == "food101":
    dataset = load_dataset("clip-benchmark/wds_food101")
elif dataset_name == "cars":
    dataset = load_dataset("clip-benchmark/wds_cars")
elif dataset_name == "country211":
    dataset = load_dataset("clip-benchmark/wds_country211")
elif dataset_name == "fgvc-aircraft":
    dataset = load_dataset("clip-benchmark/wds_fgvc_aircraft")
elif dataset_name == "cub":
    dataset = load_dataset("lxs784/cub-200-2011-clip-benchmark")

# read classes from file
classes = json.load(open(f"data/classes_{dataset_name}.json", "r"))
classes = [cls.lower().replace("_", " ") for cls in classes]

if "webp" in dataset["test"][0] and dataset["test"][0]["webp"] is not None:
    image_column = "webp"
elif dataset["test"][0]["jpg"] is not None:
    image_column = "jpg"

if train_available:
    dataset_train = dataset["train"]
    dataloader_train = DataLoader(dataset_train, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers,
                            collate_fn=lambda batch: (
                                [item[image_column] for item in batch],
                                [classes[item["cls"]] for item in batch],
                            ))
else:
    dataset_train = None
    dataloader_train = None

dataset_test = dataset["test"]
dataloader_test = DataLoader(dataset_test, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers,
                            collate_fn=lambda batch: (
                                [item[image_column] for item in batch],
                                [classes[item["cls"]] for item in batch],
                            ))

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e31')
model.to(device)
model.eval()

# encoding text
text_inputs = torch.cat([tokenizer.tokenize(f"a photo of a {c}") for c in classes]).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_inputs).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
print("Done encoding text inputs")

# encoding images
def top_predictions(image_input):
  # Calculate features
  with torch.no_grad():
      image_features = model.encode_image(image_input)

  # Pick the top 5 most similar labels for the image
  image_features /= image_features.norm(dim=-1, keepdim=True)

  # Calculate similarity
  similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
  values, indices = similarity[0].topk(5)
  return values, indices
  
# encoding images in batch
def process_batch(images):
    # images = torch.stack([preprocess(Image.open(io.BytesIO(img)).convert("RGB")) for img in images]).to(device)
    images = torch.stack([preprocess(img) for img in images]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity.topk(5)
    return values, indices

def evaluate_dataset(dataset_name, dataloader_train, dataloader_test, classes, category_name=None):
    total_images = 0
    top1 = 0
    top5 = 0

    if dataloader_train:
        for batch in tqdm(dataloader_train, desc=f"Evaluating {dataset_name} {category_name} train set"):
            image_bytes, class_names = batch
            total_images += len(image_bytes)
            value, indices = process_batch(image_bytes)

            for i, class_name in enumerate(class_names):
                predicted_classes = [classes[idx] for idx in indices[i].tolist()]
                if class_name == predicted_classes[0]:
                    top1 += 1
                if class_name in predicted_classes:
                    top5 += 1

    if dataloader_test:
        for batch in tqdm(dataloader_test, desc=f"Evaluating {dataset_name} {category_name} test set"):
            image_bytes, class_names = batch
            total_images += len(image_bytes)
            value, indices = process_batch(image_bytes)

            for i, class_name in enumerate(class_names):
                predicted_classes = [classes[idx] for idx in indices[i].tolist()]
                if class_name == predicted_classes[0]:
                    top1 += 1
                if class_name in predicted_classes:
                    top5 += 1

    if total_images == 0:
        print(f"No sample in {dataset_name} - {category_name}")
        return

    top1_accuracy = top1 / total_images * 100
    top5_accuracy = top5 / total_images * 100

    print(f"\n{dataset_name},{category_name} total images: {total_images}:", end=" ")
    print(f"Top-1: {top1_accuracy:.2f}%, {top1} images;  Top-5: {top5_accuracy:.2f}%, {top5} images.")
    return top1_accuracy, top5_accuracy

def evaluate_duplicates(dataset_name, category_name, classes):
    # print(f"\nEvaluating duplicates {category_name} in {dataset_name}")
    # train split
    result_dir_train = f"data/final/{dataset_name}-train/duplicate_categories"
    if os.path.exists(result_dir_train):
        train_ids = json.load(open(os.path.join(result_dir_train, f"{category_name}.json"), "r"))
        train_ids = [int(s.lstrip('s0')) for s in train_ids]
        new_train_set = dataset_train.select(train_ids)
        new_dataloader_train = DataLoader(new_train_set, 
                                        batch_size=batch_size, 
                                        shuffle=False, 
                                        num_workers=num_workers,
                                        collate_fn=lambda batch: (
                                            [item[image_column] for item in batch],
                                            [classes[item["cls"]] for item in batch],
                                        ))
    else:
        new_dataloader_train = None
    result_dir_test = f"data/final/{dataset_name}-test/duplicate_categories"
    test_ids = json.load(open(os.path.join(result_dir_test, f"{category_name}.json"), "r"))
    test_ids = [int(s.lstrip('s0')) for s in test_ids]
    new_test_set = dataset_test.select(test_ids)
    new_dataloader_test = DataLoader(new_test_set, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers,
                            collate_fn=lambda batch: (
                                [item[image_column] for item in batch],
                                [classes[item["cls"]] for item in batch],
                            ))
    # print(f"Evaluating {dataset_name} - {category_name}: ")
    evaluate_dataset(dataset_name, new_dataloader_train, new_dataloader_test, classes, category_name)

def evaluate_dataset_without(dataset_name, dataloader_train, dataloader_test, category_name, classes):
    print(f"\nEvaluating {dataset_name} without {category_name}")

    result_dir_train = f"data/final/{dataset_name}-train/duplicate_categories"
    result_dir_test = f"data/final/{dataset_name}-test/duplicate_categories"
    
    # Load the UIDs to exclude for train and test datasets
    category_train = set(json.load(open(os.path.join(result_dir_train, f"{category_name}.json"), "r"))) if os.path.exists(result_dir_train) else None
    category_test = set(json.load(open(os.path.join(result_dir_test, f"{category_name}.json"), "r"))) if os.path.exists(result_dir_test) else None

    total_images = 0
    top1 = 0
    top5 = 0

    def process_batch_without(image_batch, class_batch, uid_batch, category):
        nonlocal total_images, top1, top5
        filtered_indices = [i for i, uid in enumerate(uid_batch) if uid not in category]
        if not filtered_indices:
            return
        filtered_images = [image_batch[i] for i in filtered_indices]
        filtered_classes = [class_batch[i] for i in filtered_indices]

        images = torch.stack([preprocess(img) for img in filtered_images]).to(device)


        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity.topk(5)
        
        for i, class_name in enumerate(filtered_classes):
            predicted_classes = [classes[idx] for idx in indices[i].tolist()]
            if class_name == predicted_classes[0]:
                top1 += 1
            if class_name in predicted_classes:
                top5 += 1
        total_images += len(filtered_images)
    if dataloader_train:
        for batch in tqdm(dataloader_train, desc="Evaluating train dataset"):
            image_batch, class_batch, uid_batch = batch
            process_batch_without(image_batch, class_batch, uid_batch, category_train)

    if dataloader_test:
        for batch in tqdm(dataloader_test, desc="Evaluating test dataset"):
            image_batch, class_batch, uid_batch = batch
            process_batch_without(image_batch, class_batch, uid_batch, category_test)

    top1_accuracy = top1 / total_images * 100
    top5_accuracy = top5 / total_images * 100

    print(f"\n{dataset_name} without {category_name}:", end=" ")
    print(f"Top-1: {top1_accuracy:.2f}%  Top-5: {top5_accuracy:.2f}%")
    return top1_accuracy, top5_accuracy

evaluate_dataset(dataset_name, dataloader_train, dataloader_test, classes)
evaluate_dataset_without(dataset_name, dataloader_train, dataloader_test, "all_captions", classes)

evaluate_duplicates(dataset_name, "all_captions", classes)
evaluate_duplicates(dataset_name, "correct_captions", classes)
evaluate_duplicates(dataset_name, "relevant_captions", classes)
evaluate_duplicates(dataset_name, "irrelevant_captions", classes)

evaluate_duplicates(dataset_name, "only_correct", classes)
evaluate_duplicates(dataset_name, "only_relevant", classes)
evaluate_duplicates(dataset_name, "only_irrelevant", classes)
evaluate_duplicates(dataset_name, "mixed", classes)
evaluate_duplicates(dataset_name, "correct_and_relevant", classes)
evaluate_duplicates(dataset_name, "correct_and_irrelevant", classes)
evaluate_duplicates(dataset_name, "relevant_and_irrelevant", classes)