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
from tqdm import tqdm
import argparse
import requests
from glob import glob
import torchvision
import re
from transformers import set_seed
import json
from pprint import pprint
import warnings
from datasets import load_dataset
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import requests
from torch.nn.functional import log_softmax
from accelerate import Accelerator
import torch.nn.functional as F
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForImageTextToText
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
import transformers 
import torch
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer

def process_generation_llava(prompt, text, image_path, processor, model, device, with_image, img_zero, metaphor_zero):
    text = prompt.replace("REPLACE_CLAIM", text)
    text_zero = prompt.replace("REPLACE_CLAIM", "Progress through effort and determination.")
    messages = [{'role':'user', 'content':[]}]
    
    images = []
    images.append(Image.open(img_zero))
    messages[0]['content'].append({'type':'text', 'text':f'{text_zero} \n<metaphor>{metaphor_zero}</metaphor>'})
    messages[0]['content'].append({'type':'image'})

    if with_image:
        images.append(Image.open(image_path))
    messages[0]['content'].append({'type':'text', 'text':text})
    messages[0]['content'].append({'type':'image'})

    print(f'Conversation: {messages}')

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images = images, text = prompt, return_tensors="pt").to(device)

    output_ids = model.generate(**inputs, max_new_tokens=10000)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0].lower()

def process_generation_llava_without_image(prompt, implicit_meaning, processor, model, device, metaphor_zero):
    text = prompt.replace("REPLACE_CLAIM", implicit_meaning)
    text_zero = prompt.replace("REPLACE_CLAIM", "Progess through effort and determination.")
    
    messages = [{'role':'user', 'content':[]}]
    messages[0]['content'].append({'type':'text', 'text':f'{text_zero} \n<metaphor>{metaphor_zero}</metaphor>'})
    messages[0]['content'].append({'type':'text', 'text':text})

    print(f'Conversation: {messages}')

    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(device)

    output_ids = model.generate(**inputs, max_new_tokens=10000)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0].lower()

def process_generation_qwen_vision(prompt, text, image_path, processor, model, device, with_image, img_zero, metaphor_zero):
    text = prompt.replace("REPLACE_CLAIM", text)
    text_zero = prompt.replace("REPLACE_CLAIM", "Progress through effort and determination.")
    messages = [{'role':'user', 'content':[]}]

    # One shot 
    messages[0]['content'].append({'type':'image', 'image':f'file://{img_zero}'})
    messages[0]['content'].append({'type':'text', 'text':f'{text_zero} \n<metaphor>{metaphor_zero}</metaphor>'})

    if with_image:
        messages[0]['content'].append({'type':'image', 'image':f'file://{image_path}'})
    messages[0]['content'].append({'type':'text', 'text':text})

    print(f'Messages: {messages}')

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt", do_rescale=False)
    inputs = inputs.to(device)

    output_ids = model.generate(**inputs, max_new_tokens=10000)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0].lower()

def process_generation_qwen_without_image(prompt, implicit_meaning, processor, model, device, metaphor_zero):
    text = prompt.replace("REPLACE_CLAIM", implicit_meaning)
    text_zero = prompt.replace("REPLACE_CLAIM", "Progess through effort and determination.")

    messages = [{'role':'user', 'content':[]}]
    messages[0]['content'].append({'type':'text', 'text':f'{text_zero} \n<metaphor>{metaphor_zero}</metaphor>'})
    messages[0]['content'].append({'type':'text', 'text':text})

    print(f'Messages: {messages}')

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt", do_rescale=False).to(device)

    output_ids = model.generate(**inputs, max_new_tokens=10000)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0].lower()


def process_generation_qwen_text(prompt, text, tokenizer, model, device, implicit_meaning_zero, metaphor_zero):
    text_simple = prompt.replace("REPLACE_CLAIM", text)
    text_zero = prompt.replace("REPLACE_CLAIM", implicit_meaning_zero)
    text_zero_berria = f'{text_zero} \n<metaphor>{metaphor_zero}</metaphor>'
    text = f'{text_zero_berria} \n {text_simple}'

    print(f'Text: {text}')
    messages = [{'role':'user', 'content': text}]
    print(f'Messages: {messages}')

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]    
    return response

def process_generation_llama_vision(prompt_template, text, image_path, processor, model, device, with_image, img_zero, metaphor_zero):
    text = prompt_template.replace("REPLACE_CLAIM", text)
    text_zero = prompt_template.replace("REPLACE_CLAIM", "Progress through effort and determination.")
    
    image_inputs = []
    image_inputs.append(Image.open(img_zero))
    prompt = f"<|image|><|begin_of_text|>{text_zero} \n<metaphor>{metaphor_zero}</metaphor>\n"

    if with_image:
        image_inputs.append(Image.open(image_path))
        prompt += f"<|image|><|begin_of_text|>{text}"
    
    inputs = processor(image_inputs, prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=1000)
    output_text = processor.decode(output[0], skip_special_tokens=True)
    return output_text.lower()

def process_generation_llama_vision_without_image(prompt_template, text, processor, model, device, metaphor_zero):
    text = prompt_template.replace("REPLACE_CLAIM", text)
    text_zero = prompt_template.replace("REPLACE_CLAIM", "Progress through effort and determination.")

    prompt = f"<|image|><|begin_of_text|>{text_zero} \n<metaphor>{metaphor_zero}</metaphor>\n"
    prompt += f"<|image|><|begin_of_text|>{text}"

    dummy_image = Image.new("RGB", (224, 224), color=(255, 255, 255))

    print(f'Prompt: {prompt}')
    inputs = processor(images = [dummy_image, dummy_image], text = prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=1000)
    output_text = processor.decode(output[0], skip_special_tokens=True)
    return output_text.lower()


def process_generation_llama_text(prompt, implicit_meaning, tokenizer, model, device, implicit_meaning_zero, metaphor_zero):
    text_simple = prompt.replace("REPLACE_CLAIM", implicit_meaning)
    text_zero = prompt.replace("REPLACE_CLAIM", implicit_meaning_zero)
    text_zero_berria = f'{text_zero} \n<metaphor>{metaphor_zero}</metaphor>'
    text = f'{text_zero_berria} \n {text_simple}'
    print(f'Text: {text}')
    input_ids = tokenizer(text, return_tensors = 'pt').to(device)
    output = model.generate(**input_ids, max_new_tokens=1000)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response 

def process_generation_mistral(prompt, implicit_meaning, tokenizer, model, device, implicit_meaning_zero, metaphor_zero):
    text_simple = prompt.replace("REPLACE_CLAIM", implicit_meaning)
    text_zero = prompt.replace("REPLACE_CLAIM", implicit_meaning_zero)
    text_zero_berria = f'{text_zero} \n<metaphor>{metaphor_zero}</metaphor>'
    text = f'{text_zero_berria} \n {text_simple}'
    print(f'Text: {text}')

    messages = [
    {"role": "user", "content": text}]
    
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0]

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", type=str) 
    # Qwen 
    # Text only: Qwen/Qwen2.5-7B-Instruct
    # Vision: Qwen/Qwen2.5-VL-7B-Instruct

    # LLaVA
    # Text only: mistralai/Mistral-7B-Instruct-v0.2
    # Vision: llava-hf/llava-v1.6-mistral-7b-hf

    # Llama 
    # Text only: meta-llama/Llama-3.1-8B-Instruct
    # Vision: unsloth/Llama-3.2-11B-Vision-Instruct
    parser.add_argument("--type", default="text", type=str) # text or vision
    parser.add_argument("--prompt", default=3, type=int) 
    # Vision models
    # without image = 0, with image = 1
    # Texual models
    # only_text = 3
    return parser.parse_args()

def main():
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    args = config()
    accelerator = Accelerator()
    device = accelerator.device
    model_name = "" 
    print(f'Loading model...')
    if 'llava' in args.model or 'mistral' in args.model:
        if args.type == "vision":
            model_name = 'llava_vision'
            processor = LlavaNextProcessor.from_pretrained(args.model, trust_remote_code=True)
            model = LlavaNextForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        if args.type == "text":
            model_name = "llava_text"
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto", token="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")
            tokenizer = AutoTokenizer.from_pretrained(args.model, token="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")

    if 'Qwen' in args.model:
        if args.type == "vision":
            model_name = 'qwen_vision'
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
            processor = AutoProcessor.from_pretrained(args.model)
        if args.type == "text":
            model_name = 'qwen_text'
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto", token="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")
            tokenizer = AutoTokenizer.from_pretrained(args.model)

    if 'Llama' in args.model:
        if args.type == 'text':
            model_name = 'llama_text_new'
            model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16, token="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")
            tokenizer = AutoTokenizer.from_pretrained(args.model, token="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")
        if args.type == "vision":
            model_name = "llama_vision"
            model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", token="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")
            processor = AutoProcessor.from_pretrained(args.model, token="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")

    p_name = {0:'without_image', 1: 'with_image', 3:'only_text'}
    
    image_folder = "/data/ajaunarena/our_dataset/images/"
    csv_path = "/data/ajaunarena/our_dataset/our_dataset.csv"
    non_gorde = f'{p_name[args.prompt]}_{model_name}_results_fs.csv'

    print(f'Image folder: {image_folder}')
    print(f'Csv path: {csv_path}')
    print(f'Non gorde: {non_gorde}')

    if args.prompt == 0 or args.prompt == 3:
        prompt_path = "prompts/without_image_prompt_improved.txt"
    if args.prompt == 1:
        # prompt_path = "prompts/with_image_prompt.txt"
        prompt_path = "prompts/with_image_prompt_improved.txt"

    print(f'Prompt path: {prompt_path}')

    print(f'Load prompt')
    with open(prompt_path, "r") as file:
        prompt = file.read()
    print(f'Selected_prompt: {prompt}')

    img_zero = '0.jpg'
    metaphor_zero = 'She ran herself into a better situation.'
    implicit_meaning_zero = 'Progress through effort and determination.'

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['literal', 'contradiction'])
    df = df[(df['literal'].str.strip() != '') & (df['contradiction'].str.strip() != '')]
    dataset = Dataset.from_pandas(df)
    print(dataset)
    dataloader = DataLoader(dataset, batch_size = args.batch_size)
    if args.type == "vision":
        dataloader, model, processor = accelerator.prepare(dataloader, model, processor)
    if args.type == "text":
        dataloader, model, tokenizer = accelerator.prepare(dataloader, model, tokenizer)
    results = {}

    model.eval()
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        
        image_path = f'{image_folder}{batch["image_path"][0]}'
        if image_path not in results.keys():
            results[image_path] = []
            
        metaphor = batch['metaphor'][0]
        implicit_meaning = batch['literal'][0]
        print(f'Image: {image_path}')
        print(f'Implicit meaning: {implicit_meaning}')
        print()

        if args.prompt == 0: # generation without image
            if 'llava' in args.model:
                generation = process_generation_llava_without_image(prompt, implicit_meaning, processor, model, device, metaphor_zero)
            if 'Qwen' in args.model:
                generation = process_generation_qwen_without_image(prompt, implicit_meaning, processor, model, device, metaphor_zero)
            if 'Llama' in args.model: 
                generation = process_generation_llama_vision_without_image(prompt, implicit_meaning, processor, model, device, metaphor_zero)

        if args.prompt == 1: # generation with image
            if 'llava' in args.model:
                generation = process_generation_llava(prompt, implicit_meaning, image_path, processor, model, device, True, img_zero, metaphor_zero)
            if 'Qwen' in args.model:
                generation = process_generation_qwen_vision(prompt, implicit_meaning, image_path, processor, model, device, True, img_zero, metaphor_zero)
            if 'Llama' in args.model:
                generation = process_generation_llama_vision(prompt, implicit_meaning, image_path, processor, model, device, True, img_zero, metaphor_zero)

        if args.prompt == 3: # generation only with text
            if 'mistral' in args.model:
                generation = process_generation_mistral(prompt, implicit_meaning, tokenizer, model, device, implicit_meaning_zero, metaphor_zero)
            if 'Qwen' in args.model:
                generation = process_generation_qwen_text(prompt, implicit_meaning, tokenizer, model, device, implicit_meaning_zero, metaphor_zero)
            if 'llama' in args.model:
                generation = process_generation_llama_text(prompt, implicit_meaning, tokenizer, model, device, implicit_meaning_zero, metaphor_zero)
        
        print(f'True metaphor: {metaphor}')
        print(f'Generated metaphor: {generation}')
            
        results[image_path].append({
            "true_metaphor": metaphor,
            "predicted_metaphor": generation
        })

    flat_data = []
    for image_path, metaphor_entries in results.items():
        for entry in metaphor_entries:
            row = {
                    "image_path": image_path,
                    "true_metaphor": entry.get("true_metaphor", ""),
                    "predicted_metaphor": entry.get("predicted_metaphor", "")
                }
            flat_data.append(row)

    df = pd.DataFrame(flat_data)
    df.to_csv(non_gorde)

if __name__ == "__main__":
    main()

