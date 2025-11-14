import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import cohere
import argparse
import re
import itertools
import os
import time 

def prometheus(ABS_SYSTEM_PROMPT, ABSOLUTE_PROMPT, prompt, implicit_meaning, predicted_metaphor_A, predicted_metaphor_B, rubric, tokenizer, model, device, scores, all_emaitza):
        user_content = ABS_SYSTEM_PROMPT + "\n\n" + ABSOLUTE_PROMPT.format(
                orig_instruction = prompt,
                orig_response_A = predicted_metaphor_A,
                orig_response_B = predicted_metaphor_B,
                orig_reference_implicit_meaning = implicit_meaning,
                orig_criteria = rubric
        )

        messages = [
                {"role": "user", "content": user_content},
        ]

        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)

        generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        emaitza = tokenizer.batch_decode(generated_ids)[0]
        all_emaitza.append(emaitza)
        decoded = re.search(r'\[/INST\](.*)', emaitza, re.DOTALL).group(1).strip()
        # print(f'Decoded: {decoded}')
        try:
            selection = re.search(r'\[RESULT\]\s*(.*?)\s*</s>', decoded).group(1).strip()
            print(f'Selection: {selection}')
        except: 
            print(f'Incorrect prediction')
            selection = ""
        scores.append(selection)
        return all_emaitza, scores

def qwen32(ABS_SYSTEM_PROMPT, ABSOLUTE_PROMPT, prompt, implicit_meaning, predicted_metaphor_A, predicted_metaphor_B, rubric, tokenizer, model, device, scores, all_emaitza):
        user_content = ABS_SYSTEM_PROMPT + "\n\n" + ABSOLUTE_PROMPT.format(
                orig_instruction = prompt,
                orig_response_A = predicted_metaphor_A,
                orig_response_B = predicted_metaphor_B,
                orig_reference_implicit_meaning = implicit_meaning,
                orig_criteria = rubric
        )

        messages = [
                {"role": "user", "content": user_content},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        # print("thinking content:", thinking_content)
        # print("content:", content)
        all_emaitza.append(content)
        try: 
            selection = re.search(r'\[RESULT\]\s*([AB])', content).group(1)
            print(f'Selection: {selection}')
        except:
            print(f'Incorrect prediction')
            selection = ""
        scores.append(selection)
        return all_emaitza, scores

def gemma3(ABS_SYSTEM_PROMPT, ABSOLUTE_PROMPT, prompt, implicit_meaning, predicted_metaphor_A, predicted_metaphor_B, rubric, processor, model, device, scores, all_emaitza):
        user_content = ABSOLUTE_PROMPT.format(
                orig_instruction = prompt,
                orig_response_A = predicted_metaphor_A,
                orig_response_B = predicted_metaphor_B,
                orig_reference_implicit_meaning = implicit_meaning,
                orig_criteria = rubric
        )

        messages = [
                {'role':'system', 'content': [{'type':'text', 'text':ABS_SYSTEM_PROMPT}]},
                {"role": "user", "content": [{'type': 'text', 'text': user_content}]},
        ]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,return_dict=True, return_tensors="pt").to(device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
            generation = generation[0][input_len:]

        decoded = processor.decode(generation, skip_special_tokens=True)
        all_emaitza.append(decoded)
        try:
            match = re.search(r'\[RESULT\]\s*(\w+)', decoded)
            result = match.group(1).strip()
        except:
            result = ""
        print(f'Result: {result}')
        scores.append(result)
        return all_emaitza, scores

def commandA(ABS_SYSTEM_PROMPT, ABSOLUTE_PROMPT, prompt, implicit_meaning, predicted_metaphor_A, predicted_metaphor_B, rubric, co, judge_model_name, scores, all_emaitza):
        user_content = ABS_SYSTEM_PROMPT + "\n\n" + ABSOLUTE_PROMPT.format(
                orig_instruction = prompt,
                orig_response_A = predicted_metaphor_A,
                orig_response_B = predicted_metaphor_B,
                orig_reference_implicit_meaning = implicit_meaning,
                orig_criteria = rubric
        )
        full_text = ""
        for attempt in range(3):
            try:
                response = co.chat(
                        model = judge_model_name,
                        messages = [{'role':'user', 'content':user_content}]
                )
                # print(f'All: {response}')
                full_text = response.message.content[0].text
                all_emaitza.append(full_text)
                # print(f'Full text: {full_text}')
                break
            except Exception as e:
                print(f'Attempt {attempt+1} failed: {e}')
                time.sleep(5)
        else:
            raise RuntimeError(f'All attempts to get response from co.chat() failed')

        try:
            match = re.search(r'\[RESULT\]\s*(.*)', full_text)
            match = match.group(1).strip()
        except:
            match = ""
        print(f'Response: {match}')
        scores.append(match)
        return all_emaitza, scores

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model_name", default="Qwen/Qwen3-32B", type=str)
    return parser.parse_args()

def main():
    args = config()
    judge_model_name = args.judge_model_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # judge_model_name = "prometheus-eval/prometheus-7b-v2.0"
    # judge_model_name = "Qwen/Qwen3-32B"
    # judge_model_name = "Qwen/Qwen3-30B-A3B"
    # judge_model_name = "google/gemma-3-27b-it"
    # judge_model_name = "command-a-03-2025"

    if  judge_model_name == "prometheus-eval/prometheus-7b-v2.0":
        model = AutoModelForCausalLM.from_pretrained(judge_model_name)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(judge_model_name)

    if judge_model_name == "Qwen/Qwen3-32B" or judge_model_name == "Qwen/Qwen3-30B-A3B":
        tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
        model = AutoModelForCausalLM.from_pretrained(judge_model_name, torch_dtype="auto", device_map="auto")

    if judge_model_name == "google/gemma-3-27b-it": 
        model = Gemma3ForConditionalGeneration.from_pretrained(judge_model_name, device_map="auto", token="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").eval()
        processor = AutoProcessor.from_pretrained(judge_model_name, token="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")

    if judge_model_name == "command-a-03-2025":
        apiKey = "BolBckz0NKHl4OTbg4e6FR90ByFn4in5nNLyfd4B"
        co = cohere.ClientV2(apiKey)

    ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

    ABSOLUTE_PROMPT = """###Task Description:
    An instruction (might include an Input inside it), two responses to evaluate (denoted as Response A and Response B), a implicit meaning, and an evaluation criteria are given.
    1. Write a detailed feedback that assess the quality of the two responses strictly based on the given evaluation criteria, not evaluating in general.
    2. Make comparisons between Response A, Response B, and the Implicit Meaning. Instead of examining Response A and Response B separately, go straight to the point and mention about the commonalities and differences between them.
    3. After writing the feedback, indicate the better response, either "A" or "B".
    4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either "A" or "B")"
    5. Please do not generate any other opening, closing, and explanations.

    ###Instruction:
    {orig_instruction}

    ###Response A:
    {orig_response_A}

    ###Response B:
    {orig_response_B}

    ###Implicit meaning:
    {orig_reference_implicit_meaning}

    ###Score Rubric:
    {orig_criteria}

    ###Feedback:
    """

    prompt = """Compare the quality of two metaphors (Response A and Response B) and select the one that better conveys the given implicit meaning. Evaluate how closely they capture the same implicit meaning, how vivid and coherent the imagery is, and whether they effectively convey the intended concept. The goal is to determine which metaphor is a clearer, more accurate, and more imaginative representation of the original idea."""

    rubric = """5: Metaphor is highly imaginative, vivid, and aligns closely with the reference both in meaning and tone.
    4: Metaphor aligns well with the reference meaning but has slightly less vivid or original imagery.
    3: Metaphor loosely reflects the reference meaning or uses generic imagery.
    2: Metaphor has unclear or weak relation to the reference meaning, or the imagery is confusing or mixed.
    1: Metaphor does not reflect the intended meaning at all or is incoherent.
    """

    models = ['llava', 'qwen', 'llama']
    possibilities = [
            {'model_name': 'qwen_text', 'how': 'only_text'},
            {'model_name': 'qwen_vision', 'how': 'with_image'},
            {'model_name': 'qwen_vision', 'how': 'without_image'},

            {'model_name': 'llava_text', 'how': 'only_text'},
            {'model_name': 'llava_vision', 'how': 'with_image'},
            {'model_name': 'llava_vision', 'how':'without_image'},

            {'model_name': 'llama_text_new', 'how': 'only_text'},
            {'model_name': 'llama_vision', 'how': 'with_image'},
            {'model_name': 'llama_vision', 'how': 'without_image'},

            {'model_name': 'ground_truth', 'how': 'ground_truth'}
    ]
    combinations_list = list(itertools.combinations(possibilities, 2))
    combinations = []
    for combo in combinations_list:
        item1, item2 = combo
        print(f"Compare {item1['model_name']} ({item1['how']}) vs {item2['model_name']} ({item2['how']})")
        combinations.append({
            'model_name1': item1['model_name'],
            'how1': item1['how'],
            'model_name2': item2['model_name'],
            'how2': item2['how']
            })
    
    print(f'Combinations: {combinations}') 
    print(f'Number of combinations: {len(combinations)}')

    combinations_scores = {}
    for model_name in models: 
        combinations_scores[f'{model_name}_vision_with_image'] = 0
        combinations_scores[f'{model_name}_vision_without_image'] = 0
        if 'llama' in model_name: # hemen dago aldaketa
            combinations_scores[f'llama_text_new_only_text'] = 0
        else:
            combinations_scores[f'{model_name}_text_only_text'] = 0
    combinations_scores[f'ground_truth'] = 0
    
    print(f'Combinations_scores: {combinations_scores}')

    for combi in combinations: 
        model_name1 = combi['model_name1']
        model_name2 = combi['model_name2']
        how1 = combi['how1']
        how2 = combi['how2']

        print(f'Model name 1: {model_name1}, how 1: {how1}')
        print(f'Model name 2: {model_name2}, how 2: {how2}')

        if "/" in judge_model_name:
            result_name = f'{judge_model_name.split("/")[1]}_pairwise_ranking_{model_name1}_{how1}_{model_name2}_{how2}_results_fs.csv'
        else:
            result_name = f'{judge_model_name}_pairwise_ranking_{model_name1}_{how1}_{model_name2}_{how2}_results_fs.csv'

        if os.path.exists(result_name):
            print(f'Skipping {result_name} as it already exists.')
            continue

        if model_name1 != "ground_truth" and model_name2 != "ground_truth":

            without_df = pd.read_csv(f"{how1}_{model_name1}_results_fs.csv")
            without_df = without_df.drop_duplicates(['true_metaphor','image_path'])
            with_df = pd.read_csv(f"{how2}_{model_name2}_results_fs.csv")
            with_df = with_df.drop_duplicates(['true_metaphor','image_path'])
            df = pd.read_csv("/data/ajaunarena/our_dataset/our_dataset.csv")
            df = df.drop_duplicates(['metaphor', 'image_path'])
            df = df.dropna(subset=['literal', 'contradiction'])
            df = df[(df['literal'].str.strip() != '') & (df['contradiction'].str.strip() != '')]
            df = df[['metaphor', 'image_path', 'implicit_meaning']]

            scores = []
            all_emaitza = []

            for idx, row in without_df.iterrows():
                true_metaphor = row["true_metaphor"]
                implicit_meaning = df[df['metaphor']==true_metaphor]['implicit_meaning'].item()
                predicted_metaphor_A = row["extracted_metaphor"]
                predicted_metaphor_B = with_df[with_df['true_metaphor'] == true_metaphor]['extracted_metaphor'].item()

                print(f'Implicit meaning: {implicit_meaning}')
                print(f'Predicted metaphor A: {predicted_metaphor_A}')
                print(f'Predicted metaphor B: {predicted_metaphor_B}')

                if judge_model_name == "prometheus-eval/prometheus-7b-v2.0":
                    all_emaitza, scores = prometheus(ABS_SYSTEM_PROMPT, ABSOLUTE_PROMPT, prompt, implicit_meaning, predicted_metaphor_A, predicted_metaphor_B, rubric, tokenizer, model, device, scores, all_emaitza)

                if judge_model_name == "Qwen/Qwen3-32B" or judge_model_name == "Qwen/Qwen3-30B-A3B":
                    all_emaitza, scores = qwen32(ABS_SYSTEM_PROMPT, ABSOLUTE_PROMPT, prompt, implicit_meaning, predicted_metaphor_A, predicted_metaphor_B, rubric, tokenizer, model, device, scores, all_emaitza)

                if judge_model_name == "google/gemma-3-27b-it":
                    all_emaitza, scores = gemma3(ABS_SYSTEM_PROMPT, ABSOLUTE_PROMPT, prompt, implicit_meaning, predicted_metaphor_A, predicted_metaphor_B, rubric, processor, model, device, scores, all_emaitza)

                if judge_model_name == "command-a-03-2025": 
                    all_emaitza, scores = commandA(ABS_SYSTEM_PROMPT, ABSOLUTE_PROMPT, prompt, implicit_meaning, predicted_metaphor_A, predicted_metaphor_B, rubric, co, judge_model_name, scores, all_emaitza)

            # print(f'Scores: {scores}')
            result_df = pd.DataFrame()
            result_df['predicted_metaphor_A'] = without_df['extracted_metaphor'].tolist()
            result_df['predicted_metaphor_B'] = with_df['extracted_metaphor'].tolist()
            result_df[f"{judge_model_name}_score"] = scores
            result_df[f'{judge_model_name}_generation'] = all_emaitza
            result_df.to_csv(result_name, index=False)

            count_a = scores.count('A')
            count_b = scores.count('B')
            
            if count_a > count_b:
                print()
                print(f'{model_name1}_{how1} wins')
                combinations_scores[f'{model_name1}_{how1}']+=1
            elif count_b > count_a:
                print()
                print(f'{model_name2}_{how2} wins')
                combinations_scores[f'{model_name2}_{how2}']+=1
            else:
                print()
                print(f'Tie between {model_name1}_{how1} and {model_name2}_{how2}')
            print()
        else: 

            df = pd.read_csv("/data/ajaunarena/our_dataset/our_dataset.csv")
            df = df.drop_duplicates(['metaphor', 'image_path'])
            df = df.dropna(subset=['literal', 'contradiction'])
            df = df[(df['literal'].str.strip() != '') & (df['contradiction'].str.strip() != '')]
            df = df[['metaphor', 'image_path', 'implicit_meaning']]

            if model_name1 == 'ground_truth':
                df_how = pd.read_csv(f'{how2}_{model_name2}_results_fs.csv')
                df_how = df_how.drop_duplicates(['true_metaphor','image_path'])
            else:
                df_how = pd.read_csv(f'{how1}_{model_name1}_results_fs.csv')
                df_how = df_how.drop_duplicates(['true_metaphor','image_path'])

            scores = []
            all_emaitza = []

            for idx, row in df_how.iterrows():
                predicted_metaphor_A = row['true_metaphor']
                predicted_metaphor_B = row['extracted_metaphor']
                implicit_meaning = df[df['metaphor']==predicted_metaphor_A]['implicit_meaning'].item()

                print(f'Implicit meaning: {implicit_meaning}')
                print(f'Predicted metaphor A: {predicted_metaphor_A}')
                print(f'Predicted metaphor B: {predicted_metaphor_B}')

                if judge_model_name == "prometheus-eval/prometheus-7b-v2.0":
                    all_emaitza, scores = prometheus(ABS_SYSTEM_PROMPT, ABSOLUTE_PROMPT, prompt, implicit_meaning, predicted_metaphor_A, predicted_metaphor_B, rubric, tokenizer, model, device, scores, all_emaitza)

                if judge_model_name == "Qwen/Qwen3-32B" or judge_model_name == "Qwen/Qwen3-30B-A3B":
                    all_emaitza, scores = qwen32(ABS_SYSTEM_PROMPT, ABSOLUTE_PROMPT, prompt, implicit_meaning, predicted_metaphor_A, predicted_metaphor_B, rubric, tokenizer, model, device, scores, all_emaitza)

                if judge_model_name == "google/gemma-3-27b-it":
                    all_emaitza, scores = gemma3(ABS_SYSTEM_PROMPT, ABSOLUTE_PROMPT, prompt, implicit_meaning, predicted_metaphor_A, predicted_metaphor_B, rubric, processor, model, device, scores, all_emaitza)

                if judge_model_name == "command-a-03-2025":
                    all_emaitza, scores = commandA(ABS_SYSTEM_PROMPT, ABSOLUTE_PROMPT, prompt, implicit_meaning, predicted_metaphor_A, predicted_metaphor_B, rubric, co, judge_model_name, scores, all_emaitza)

            # print(f'Scores: {scores}')
            result_df = pd.DataFrame()
            result_df['predicted_metaphor_A'] = df_how['true_metaphor'].tolist()
            result_df['predicted_metaphor_B'] = df_how['extracted_metaphor'].tolist()
            result_df[f"{judge_model_name}_score"] = scores
            result_df[f'{judge_model_name}_generation'] = all_emaitza
            result_df.to_csv(result_name, index=False)

            count_a = scores.count('A')
            count_b = scores.count('B')

            if count_a > count_b:
                print()
                print(f'ground truth wins')
                combinations_scores[f'ground_truth']+=1
            elif count_b > count_a:
                print()
                if model_name1 != 'ground_truth':
                    print(f'{model_name1}_{how1} wins')
                    combinations_scores[f'{model_name1}_{how1}']+=1
                else:
                    print(f'{model_name2}_{how2} wins')
                    combinations_scores[f'{model_name2}_{how2}']+=1 
            else:
                print()
                print(f'Tie between {model_name1}_{how1} and {model_name2}_{how2}')
            print()

        print(f'Combinations: {combinations_scores}')
    print(f'Combinations: {combinations_scores}')

if __name__ == "__main__":
    main()

