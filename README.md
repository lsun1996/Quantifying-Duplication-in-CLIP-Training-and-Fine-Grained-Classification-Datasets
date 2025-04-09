This repository contains the research code for our paper "Quantifying Duplication in CLIP Training and Fine-Grained Classification Datasets".

# Key Features

2. Analyze captions of the duplicate images, categorize the duplicates into exact match, relevant captions, and irrelevant captions;

3. Get zero-shot accuracy on original dataset and duplicates with Open CLIP.

To reproduce results in the paper, download the json files from drive and put into data/final folder, the results are manually verified and used to produce table:

# Instructions

The scirpts need to run in lightning studio due to the data in laion400m is shared within lightning.ai community. Registration is required.

<a target="_blank" href="https://lightning.ai/lesun-jjxwd/studios/quantifying-duplication-in-clip-training-and-fine-grained-classification-datasets">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>

After opening the studio

1. Install dependencies:
```
pip install -r requirements.txt​
```

2. Find duplicates in target dataset.

Specify dataset name in the "Parameter Configuration" section on top of the script, enter OpenAI API key to enable caption analysis.
```
python main.py​
```
3. Zero-shot Testing 

Specify dataset name in the "Parameter Configuration" section on top of the script.
```
python zero_shot.py​`\
```
