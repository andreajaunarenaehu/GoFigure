from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
import PIL
import hashlib
from collections import Counter
from torchvision import transforms
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
from collections import defaultdict
import ast
import re
from accelerate import Accelerator
import torch

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--model", default="openai/clip-vit-base-patch16", type=str)
    return parser.parse_args()

def main():
    torch.cuda.empty_cache()
    print(torch.cuda.device_count())
    args = config()
    accelerator = Accelerator()
    device = accelerator.device

    # model, preprocess = clip.load(args.model, device = device)
    model = CLIPModel.from_pretrained(args.model)
    processor = CLIPProcessor.from_pretrained(args.model)
    print(f'Model loaded')

    image_folder = "/data/ajaunarena/our_dataset/images"
    csv_path = "/data/ajaunarena/our_dataset/ImageMet_dataset.csv"
    gorde = f"ImageMet_nli_original_all_clip.csv"

    df = pd.read_csv(csv_path)
    print(f'ImageMet shape: {df.shape}')
    dataset = Dataset.from_pandas(df)
    print(dataset)

    dataloader = DataLoader(dataset, batch_size = args.batch_size)

    dataloader, model = accelerator.prepare(dataloader, model)

    total = 0
    acc = 0
    entailment_metaphor_list = []
    contradiction_metaphor_list = []
    image_list = []
    score_e_list = []
    score_c_list = []

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        image_path = f'{image_folder}/{batch["image_path"][0]}'
        entailment_metaphor = batch['entailment metaphor'][0]
        contradiction_metaphor = batch['contradiction metaphor'][0]
        print(f'Image: {image_path}')
        print(f'Entailment metaphor: {entailment_metaphor}')
        print(f'Contradiction mentaphor: {contradiction_metaphor}')
        
        image = Image.open(image_path)

        # Entailment metaphor
        inputs_e = processor(text=[entailment_metaphor], images=image, return_tensors="pt", padding=True).to(device)
        outputs_e = model(**inputs_e)
        logits_e = outputs_e.logits_per_image
        generation_entailment = logits_e.detach().cpu().item()
        print(f'Probs: {generation_entailment}')

        # Contradiction metaphor 
        inputs_c = processor(text=[contradiction_metaphor], images=image, return_tensors="pt", padding=True).to(device)
        outputs_c = model(**inputs_c)
        logits_c = outputs_c.logits_per_image
        generation_contradiction = logits_c.detach().cpu().item()
        print(f'Probs: {generation_contradiction}')

        if generation_entailment > generation_contradiction:
            acc += 1
        total += 1

        entailment_metaphor_list.append(entailment_metaphor)
        contradiction_metaphor_list.append(contradiction_metaphor)
        image_list.append(image_path)
        score_e_list.append(generation_entailment)
        score_c_list.append(generation_contradiction)
            
        print(f'Accuracy total: {acc}/{total} --> {(acc/total)*100:.2f}%')
 
    df = pd.DataFrame()
    df['entailment metaphor'] = entailment_metaphor_list
    df['contradiction metaphor'] = contradiction_metaphor_list
    df['image'] = image_list
    df['score entailment metaphor'] = score_e_list
    df['score contradiction metaphor'] = score_c_list
    df.to_csv(gorde)

if __name__ == "__main__":
    main()

