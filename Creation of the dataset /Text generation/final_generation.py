import os
import re
import pandas as pd
import anthropic
import time

API_KEY = "" # write here you anthropic key 
client = anthropic.Anthropic(
    api_key=API_KEY,
)

df = pd.read_csv("source_target_final.csv")
aukeraketa = 2

# If you want to generate entailing literals and metaphors and contradicting literals from mappings (source and target domains), run aukeraketa = 1

# Generated entailing metaphors and literal and contradicting literals will be loaded in this directory. 
# If you have an error while loading the code, you can run the code again an already generated instances will not be generated again. 
directory = "generations/metaphors_generated/"

# Prompt to generate entailing metaphors, entailing literals and contradicting literals
with open("hirurak_prompt.txt", "r") as file: 
    first_prompt = file.read()

print(f'Prompt to generate metaphors: {first_prompt}')

for index, row in df.iterrows():
	if aukeraketa == 1:
		source = row['source']
		target = row['target']
		print(f'Source: {source}')
		print(f'Target: {target}')
		# Generate 10 metaphors per source target pairs
		first_prompt2 = first_prompt.replace("{{SOURCE}}", source).replace("{{TARGET}}", target)
		message = client.messages.create(model = "claude-3-5-sonnet-20241022", max_tokens = 1000, temperature = 0, messages = [{ "role" : "user", "content" : [{ "type" : "text", "text" : first_prompt2}]}])
		first_result = message.content[0].text
		print(f'First result: {first_result}')
		with open(directory+source+"_"+target+".txt", "w") as file: 
			file.write(first_result)

# If you want to generate visual elaborations form linguistic metaphors, run aukeraketa = 2

# Generated visual elaborations will be loaded in this directory. 
# If you have an error while loading the code, you can run the code again an already generated instances will not be generated again. 
second_directory = "generations/visual_elaboration/"

# Generated entailing metaphors and literal and contradicting literals can be found in this csv file. 
first = "generations/metaphors_generated/first_step.csv"

# Prompt taken from HAIVMet dataset (https://arxiv.org/abs/2305.14724)
with open("ISM_modified.txt", "r") as file: 
    prompt_template = file.read()

# Examples given in HAIVMet dataset (https://arxiv.org/abs/2305.14724)
with open("ISM_few_shot_examples.txt", "r") as file:
	examples = file.read()

print(f'ISM modified prompt: {prompt_template}')
print(f'Examples: {examples}')

if aukeraketa == 2:
	a = pd.read_csv(first)
	for index, row in a.iterrows():
		metaphor = row['metaphor']
		print(f'Index: {index}, metaphor: {metaphor}')
		path = os.path.join(second_directory,f'{str(index)}.txt')
		if os.path.isfile(path):
			print(f'The file {str(index)} and metaphor {metaphor} exists in the directory')
		else:
			print(f'The file {str(index)} and metaphor {metaphor} does not exist in the directory')
			final_prompt = prompt_template.replace("{{LINGUISTIC_METAPHOR}}", metaphor)
			message = client.messages.create(model = "claude-3-5-sonnet-20241022", max_tokens = 1000, temperature = 0, messages = [{"role" : "user","content" : [{"type": "text","text": examples},{"type" : "text","text" : final_prompt}]}])
			result = message.content[0].text
			print(result)
			print(f'Done!')
			print()
			with open(second_directory+str(index)+".txt", "w") as file: 
				file.write(result)
			with open(second_directory+str(index)+".txt", "r") as file:
				print(file.readlines())
