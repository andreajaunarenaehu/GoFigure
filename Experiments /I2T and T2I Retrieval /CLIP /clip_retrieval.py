from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
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

def hash_image(image):
    return hashlib.md5(image.tobytes()).hexdigest() 

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--task", default="retrieval_t2i", type=str) # retrieval_i2t, retrieval_t2i
    parser.add_argument("--image_folder", default="/data/ajaunarena/our_dataset/images", type=str)
    parser.add_argument("--csv_path", default="/data/ajaunarena/our_dataset/our_dataset.csv", type=str)
    parser.add_argument("--gorde", default="results_our_dataset_retrieval_t2i_16_patch.csv", type=str)
    return parser.parse_args()

def main():
    torch.cuda.empty_cache()
    print(torch.cuda.device_count())
    args = config()
    accelerator = Accelerator()
    device = accelerator.device

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    print(f'Model loaded: {args.model}')

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

        dataloader, model = accelerator.prepare(dataloader, model)

        i2t_results = {}
        i2t_acc = 0

        for batch_idx, example in tqdm(enumerate(dataloader), total = len(dataloader)):
            print(f'Instance: {example}')
            image_path = f'{args.image_folder}/{example["image_path"][0]}'
            if image_path not in i2t_results.keys():
                i2t_results[image_path] = []
            image = Image.open(image_path)
            metaphors = example['metaphor'][0]
            print(f'Image: {image_path}')
            print(f'Metaphor: {metaphors}')
            print()
            # Process texts
            probs = []
            for m_idx, metaphor in enumerate(unique_metaphors):
                met = metaphor
                inputs = processor(text=[metaphor], images=image, return_tensors="pt", padding=True).to(device)
                outputs = model(**inputs)

                logits_per_image = outputs.logits_per_image
                logits_per_image = logits_per_image.detach().cpu().item()
                # print(f'Metaphor: {met}, logits: {logits_per_image}')
                probs.append(logits_per_image)
                i2t_results[image_path].append({met:logits_per_image})
       
            max_value = max(probs)
            maximoak = [i for i, val in enumerate(probs) if val == max_value]
            for maximoa_idx in maximoak:
                print(f'Predicted index: {maximoa_idx}')
                maximoa_metaphor = unique_metaphors[maximoa_idx]
                print(f'Predicted metaphor: {maximoa_metaphor}')
                if maximoa_metaphor in metaphors:
                    i2t_acc+=1
                    print(f'Acc: {i2t_acc}')
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
        print(dataset[0])

        unique_metaphors = dataset['metaphor']
        print(f'Unique metaphors: unique_metaphors, size: {len(unique_metaphors)}')
        unique_images = dataset['image_path']
        print(f'Unique images: {unique_images}, size: {len(unique_metaphors)}')

        dataloader = DataLoader(dataset, batch_size = args.batch_size)

        dataloader, model = accelerator.prepare(dataloader, model)
        
        t2i_results = {}
        t2i_acc = 0

        for index, example in tqdm(enumerate(dataloader), total = len(dataloader)):
            print(f'Instance: {example}')
            images = example['image_path'][0]
            metaphor = example['metaphor'][0]
            if metaphor not in t2i_results.keys():
                t2i_results[metaphor] = []
            print(f'Image: {images}, size: {len(images)}')
            print(f'Metaphor: {metaphor}')
            print()

            probs = []
            # Process texts
            for i_idx, image_path in enumerate(unique_images):
                image_path = f'{args.image_folder}/{image_path}'
                image = Image.open(image_path)
                inputs = processor(text=[metaphor],images=image,return_tensors="pt",padding=True).to(device)
                outputs = model(**inputs)
                logits_per_text = outputs.logits_per_image
                logits_per_text = logits_per_text.detach().cpu().item()
                probs.append(logits_per_text)
                t2i_results[metaphor].append({image_path:logits_per_text})
                # print(f'Logits per image: {logits_per_text}')

            max_value = max(probs)
            maximoak = [i for i, val in enumerate(probs) if val == max_value]
            for maximoa_idx in maximoak:
                print(f'Predicted index: {maximoa_idx}')
                maximoa_image = unique_images[maximoa_idx]
                print(f'Predicted image: {maximoa_image}')
                if maximoa_image in images:
                    t2i_acc+=1
                    print(f'Acc: {t2i_acc}')
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

