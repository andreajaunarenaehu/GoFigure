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
from tqdm import tqdm
import argparse
import requests
from transformers import AutoProcessor, AutoModel
from transformers import SiglipProcessor, SiglipTokenizer
import ast 
from accelerate import Accelerator

# Function to hash image content
def hash_image(image):
    return hashlib.md5(image.tobytes()).hexdigest()  

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--model", default="google/siglip-base-patch16-224", type=str)
    parser.add_argument("--task", default="retrieval_t2i", type=str) # retrieval_i2t, retrieval_t2i
    parser.add_argument("--image_folder", default="/data/ajaunarena/our_dataset/images/", type=str)
    parser.add_argument("--csv_path", default="/data/ajaunarena/our_dataset/our_dataset.csv", type=str)
    parser.add_argument("--gorde", default="results_retrieval_t2i_our_dataset.csv", type=str)
    return parser.parse_args()

def main():
    args = config()
    accelerator = Accelerator()
    device = accelerator.device
    
    print(f'Loading model...')
    model = AutoModel.from_pretrained(args.model, device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(args.model)

    print(f'Model loaded successfully!')

    # I2T
    if args.task == "retrieval_i2t":

        df = pd.read_csv(args.csv_path)
        df = df[['image_path', 'metaphor']]
        dataset = Dataset.from_pandas(df)
        print(dataset)
        unique_images = dataset['image_path']
        print(f'Unique images: {len(unique_images)}')
        unique_metaphors = dataset['metaphor']
        print(f'Unique metaphors: {len(unique_metaphors)}')
        dataloader = DataLoader(dataset, batch_size = args.batch_size)

        dataloader, model, processor = accelerator.prepare(dataloader, model, processor) 

        i2t_results = {}
        i2t_acc = 0

        for index, example in tqdm(enumerate(dataloader), total=len(dataloader)):
            image_path = f'{args.image_folder}/{example["image_path"][0]}'
            if image_path not in i2t_results.keys():
                i2t_results[image_path] = []
            image = Image.open(image_path).convert("RGB")

            metaphors = example['metaphor'][0]
            print(f'Image: {image}')
            print(f'Metaphor: {metaphors}, len: {len(metaphors)}')
            # Process texts
            probs_list = []
            for m_idx, metaphor in enumerate(unique_metaphors):
                inputs = processor(text=[metaphor], images=image, padding=True, truncation=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = torch.sigmoid(logits_per_image)[0][0].detach().cpu()
                probs_list.append(probs)
                i2t_results[image_path].append({metaphor:probs})

            max_value = max(probs_list)
            maximoak = [i for i, val in enumerate(probs_list) if val == max_value]
            for maximoa_idx in maximoak:
                print(f'Predicted index: {maximoa_idx}')
                maximoa_metaphor = unique_metaphors[maximoa_idx]
                print(f'Predicted metaphor: {maximoa_metaphor}')
                if maximoa_metaphor in metaphors:
                    i2t_acc+=1
                    print(f'Acc: {i2t_acc}')
                    print()
                    break
            print(f'Acc: {i2t_acc}/{len(unique_images)} --> {(i2t_acc/len(unique_images))*100:.2f}%')
        print(f'Acc: {i2t_acc}/{len(unique_images)} --> {(i2t_acc/len(unique_images))*100:.2f}%')
        flat_results = []
        
        for image_path, predictions in i2t_results.items():
            for pred in predictions:
                for metaphor, score in pred.items():
                    flat_results.append({
                        "image_path": image_path,
                        "metaphor": metaphor,
                        "score": score
                    })
                    
        results_df = pd.DataFrame(flat_results)
        results_df.to_csv(args.gorde)

    # T2I
    if args.task == "retrieval_t2i":

        df = pd.read_csv(args.csv_path)
        df = df[['image_path', 'metaphor']]
        dataset = Dataset.from_pandas(df)
        print(dataset)

        unique_metaphors = dataset['metaphor']
        print(f'Unique metaphors: {unique_metaphors}, size: {len(unique_metaphors)}')
        unique_images = dataset['image_path']
        print(f'Unique images: {unique_images}, size: {len(unique_metaphors)}')

        dataloader = DataLoader(dataset, batch_size = args.batch_size)

        dataloader, model, processor = accelerator.prepare(dataloader, model, processor)

        t2i_results = {}
        t2i_acc = 0

        for index, example in tqdm(enumerate(dataloader), total=len(dataloader)):
            images = example['image_path'][0]
            metaphor = example['metaphor'][0]
            if metaphor not in t2i_results.keys():
                t2i_results[metaphor] = []
            print(f'Image: {images}, size: {len(images)}')
            print(f'Metaphor: {metaphor}')
            # Process texts
            probs_list = []
            for i_idx, image_path in enumerate(unique_images):
                image_path = f'{args.image_folder}/{image_path}'
                image = Image.open(image_path).convert("RGB")
                inputs = processor(text=[metaphor], images=image, padding=True, truncation=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = torch.sigmoid(logits_per_image)[0][0].detach().cpu()
                probs_list.append(probs)
                t2i_results[metaphor].append({image_path:probs})

            max_value = max(probs_list)
            maximoak = [i for i, val in enumerate(probs_list) if val == max_value]
            for maximoa_idx in maximoak:
                maximoa_image = unique_images[maximoa_idx]
                print(f'Predicted image: {maximoa_image}')
                if maximoa_image in images:
                    t2i_acc+=1
                    print(f'Acc: {t2i_acc}')
                    print()
                    break
            print(f'Acc: {t2i_acc}/{len(unique_metaphors)} --> {(t2i_acc/len(unique_metaphors))*100:.2f}%')
        print(f'Acc: {t2i_acc}/{len(unique_metaphors)} --> {(t2i_acc/len(unique_metaphors))*100:.2f}%')

        flat_results = []
        
        for metaphor, predictions in t2i_results.items():
            for pred in predictions:
                for image_path, score in pred.items():
                    flat_results.append({
                        "metaphor": metaphor,
                        "image_path":image_path,
                        "score": score
                    })


        results_df = pd.DataFrame(flat_results)
        results_df.to_csv(args.gorde)

if __name__ == "__main__":
    main()

