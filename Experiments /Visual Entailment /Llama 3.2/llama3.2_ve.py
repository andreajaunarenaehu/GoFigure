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
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
from collections import defaultdict
import ast
import re
from accelerate import Accelerator
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForImageTextToText
import torch.nn.functional as F

def process_generation_with_image(prompt_template, text, image_path, processor, model, device):
    text = prompt_template.replace("REPLACE_CLAIM", text)
    print(f'Prompt: {text}')
    image_inputs = []
    image_inputs.append(Image.open(image_path))
    prompt = f"<|image|><|begin_of_text|>{text}"
    inputs = processor(image_inputs, prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=100)
    output_text = processor.decode(output[0], skip_special_tokens=True)
    if output_text.startswith(text):
        output_text = output_text[len(text):].strip()
    return output_text.lower()

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--model", default="unsloth/Llama-3.2-11B-Vision-Instruct", type=str) 
    parser.add_argument("--prompt", default=0, type=int) # nli= 0 or qa=1 or chain=2 or improved_chain=3
    return parser.parse_args()

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.empty_cache()
    print(torch.cuda.device_count())
    args = config()

    accelerator = Accelerator()
    device = accelerator.device

    print(f'Loading model...')
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", token="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")
    processor = AutoProcessor.from_pretrained(args.model, token="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")

    print(f'Model loaded: {args.model}')

    p_names = {0:'nli', 1:'vqa', 2:'chain', 3:'chain_improved'}

    image_folder = "/data/ajaunarena/our_dataset/images"
    csv_path = "/data/ajaunarena/our_dataset/ImageMet_dataset.csv"
    non_gorde = f'{p_names[args.prompt]}_ImageMet_all2.csv'

    print(f'Image folder: {image_folder}')
    print(f'Csv path: {csv_path}')
    print(f'Non gorde: {non_gorde}')
    
    if args.prompt == 0:
        prompt_path = "prompts/nli_zero_prompt.txt"
    if args.prompt == 1:
        prompt_path = "prompts/qa_zero_prompt.txt"

    print(f'Prompt path: {prompt_path}')

    print(f'Load prompt')
    with open(prompt_path, "r") as file:
        prompt = file.read()
    print(f'Selected_prompt: {prompt}')

    if "nli" in prompt_path: 
        entailment_key = {"entailment", "entails", "entail"}
        contradiction_key = {"contradiction", "contradicts", "contradicted", "contradictory", "contradict"}
        entailment_label = "entailment"
        contradiction_label = "contradiction"

    if ("qa" in prompt_path) or ("chain" in prompt_path):
        entailment_key = {"yes"}
        contradiction_key = {"no"}
        entailment_label = "yes"
        contradiction_label = "no"
    
    df = pd.read_csv(csv_path)
    print(f'ImageMet shape: {df.shape}')
    dataset = Dataset.from_pandas(df)
    print(dataset)

    dataloader = DataLoader(dataset, batch_size = args.batch_size)
    dataloader, model, processor = accelerator.prepare(dataloader, model, processor)

    acc = 0
    acc_norm = 0
    total = 0
    total_e = 0
    acc_c = 0
    acc_e = 0
    acc_m = 0
    acc_m_c = 0
    lc = 0
    le = 0
    entailment_metaphor = ""
    contradiction_metaphor = ""
    results = {}

    output_df = pd.read_csv(f"{p_names[args.prompt]}_ImageMet_all.csv")
        
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        image_path = f'{image_folder}/{batch["image_path"][0]}'
        if image_path not in results.keys():
            results[image_path] = []
            
        entailment_metaphor = batch['entailment metaphor'][0]
        contradiction_metaphor = batch['contradiction metaphor'][0]
        literal_meaning = batch['literal meaning'][0]
        contradiction_meaning = batch['contradiction meaning'][0]
        total += 4
        total_e += 1

        print(f'Image: {image_path}')
        print(f'Entailment metaphor: {entailment_metaphor}')
        print(f'Contradiction mentaphor: {contradiction_metaphor}')
        print(f'Literal meaning: {literal_meaning}')
        print(f'Contradiction meaning: {contradiction_meaning}')

        if entailment_metaphor not in output_df['metaphor'].tolist():
            generation = process_generation_with_image(prompt, entailment_metaphor, image_path, processor, model, device)
            print(f'Generation entailment metaphor: {generation}')
            print()

            if any(entailment in generation for entailment in entailment_key):
                acc+=1
                acc_m+=1

            results[image_path].append({
                'metaphor': entailment_metaphor,
                'true': entailment_label,
                'prediction': generation
            })
        else:
            print(f'Generation entailment metaphor already done: {output_df[output_df["metaphor"]==entailment_metaphor]["prediction"].item()}')
            results[image_path].append({
                'metaphor': entailment_metaphor,
                'true': entailment_label,
                'prediction': output_df[output_df['metaphor']==entailment_metaphor]['prediction'].item()
            })

        if contradiction_metaphor not in output_df['metaphor'].tolist():
            generation = process_generation_with_image(prompt, contradiction_metaphor, image_path, processor, model, device)
            print(f'Generation contradiction metaphor: {generation}')
            print()

            if any(contradiction in generation for contradiction in contradiction_key):
                acc+=1
                acc_m_c+=1

            results[image_path].append({
                'metaphor': contradiction_metaphor,
                'true': contradiction_label,
                'prediction': generation
            })
        else:
            print(f'Generation contradiction metaphor already done: {output_df[output_df["metaphor"]==contradiction_metaphor]["prediction"].item()}')
            results[image_path].append({
                'metaphor': contradiction_metaphor,
                'true': contradiction_label,
                'prediction': output_df[output_df['metaphor']==contradiction_metaphor]['prediction'].item()
            })

        if literal_meaning not in output_df['metaphor'].tolist():
            generation_literal = process_generation_with_image(prompt, literal_meaning, image_path, processor, model, device)
            print(f'Generation literal meaning: {generation_literal}')
            print()

            if any(entailment in generation_literal for entailment in entailment_key):
                acc+=1
                acc_e+=1

            results[image_path].append({
                'metaphor': literal_meaning,
                'true': entailment_label,
                'prediction': generation_literal
            })
        else:
            print(f'Generation literal meaning already done: {output_df[output_df["metaphor"]==literal_meaning]["prediction"].item()}')
            results[image_path].append({
                'metaphor': literal_meaning,
                'true': entailment_label,
                'prediction': output_df[output_df['metaphor']==literal_meaning]['prediction'].item()
            })

        if contradiction_meaning not in output_df['metaphor'].tolist():
            generation_contradiction = process_generation_with_image(prompt, contradiction_meaning, image_path, processor, model, device)
            print(f'Generation contradiction meaning: {generation_contradiction}')
            print()

            if any(contradiction in generation_contradiction for contradiction in contradiction_key):
                acc+=1
                acc_c+=1

            results[image_path].append({
                'metaphor': contradiction_meaning,
                'true': contradiction_label,
                'prediction': generation_contradiction
            })
        else:
            print(f'Generation contradiction meaning already done: {output_df[output_df["metaphor"]==contradiction_meaning]["prediction"].item()}')
            results[image_path].append({
                'metaphor': contradiction_meaning,
                'true': contradiction_label,
                'prediction': output_df[output_df['metaphor']==contradiction_meaning]['prediction'].item()
            })

        print(f'Accuracy total: {acc}/{total} --> {(acc/total)*100:.2f}%')
        print(f'Accuracy entailment metaphor: {acc_m}/{total_e} --> {(acc_m/total_e)*100:.2f}%')
        print(f'Accuracy contradiction metaphor: {acc_m_c}/{total_e} --> {(acc_m_c/total_e)*100:.2f}%')
        print(f'Accuracy literal meaning: {acc_e}/{total_e} --> {(acc_e/total_e)*100:.2f}%')
        print(f'Accuracy contradiction meaning: {acc_c}/{total_e} --> {(acc_c/total_e)*100:.2f}%')
        print()

    print(f'Accuracy total: {acc}/{total} --> {(acc/total)*100:.2f}%')
    print(f'Accuracy entailment metaphor: {acc_m}/{total_e} --> {(acc_m/total_e)*100:.2f}%')
    print(f'Accuracy contradiction metaphor: {acc_m_c}/{total_e} --> {(acc_m_c/total_e)*100:.2f}%')
    print(f'Accuracy literal meaning: {acc_e}/{total_e} --> {(acc_e/total_e)*100:.2f}%')
    print(f'Accuracy total: {acc_c}/{total_e} --> {(acc_c/total_e)*100:.2f}%')
    flat_data = []
    for image_path, metaphor_entries in results.items():
        for entry in metaphor_entries:
            row = {
                    "image_path": image_path,
                    "metaphor": entry.get("metaphor", ""),
                    "true": entry.get("true", ""),
                    "prediction": entry.get("prediction", "")
                }
            flat_data.append(row)

    df = pd.DataFrame(flat_data)
    df.to_csv(non_gorde)


if __name__ == "__main__":
    main()

