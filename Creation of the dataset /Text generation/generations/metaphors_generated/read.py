import csv
import re
import os
import pandas as pd
from collections import Counter

directory_path = '/scratch/ajaunarena/claude-env/final_prompt/metaphors_generated/'

items = os.listdir(directory_path)
print(len(items))
metaphors = []
contras = []
literals = []
sources = []
targets = []

metaphor_pattern = r"<metaphor>(.*?)</metaphor>"
contradiction_pattern = r"<contradiction_paraphrase>(.*?)</contradiction_paraphrase>"
literal_pattern = r"<literal_paraphrase>(.*?)</literal_paraphrase>"
for z_item, item in enumerate(items):
	# print(f'Item number:{z_item}')
	try:
		if ".txt" in item:
			# print(item)
			source_target = item.split(".")[0]
			if "_" in source_target:
				# print(f'Source target pair: {source_target}')
				# source = source_target.split("_")[0]
				# print(f'Source: {source}')
				# target = source_target.split("_")[1]
				# print(f'Target: {target}')
				source = "_".join(source_target.split("_")[:-1])  # Everything before the last underscore
				target = source_target.split("_")[-1]  # The last segment after the last underscore
			else: 
				raise ValueError(f"Unexpected file name format: {item}")
			with open(os.path.join(directory_path, item), 'r') as file:
				lines = " ".join(file.readlines())
				metaphor = re.findall(metaphor_pattern, lines, re.DOTALL)
				contradiction = re.findall(contradiction_pattern, lines, re.DOTALL)
				literal = re.findall(literal_pattern, lines, re.DOTALL)
				"""for i in range(len(metaphor)):
					metaphors.append(metaphor[i].strip())
					sources.append(source)
					targets.append(target)
					literals.append(literal[i].strip())
					contras.append(contradiction[i].strip())
				"""
				if len(metaphor) == len(contradiction) == len(literal):
					for i in range(len(metaphor)):
						metaphors.append(metaphor[i].strip())
						contras.append(contradiction[i].strip())
						sources.append(source)
						targets.append(target)
						literals.append(literal[i].strip())
				else:
					# raise ValueError(f"Mismatch in lengths of metaphor, contradiction, and literal for file: {item}, {len(metaphor)}, {len(contradiction)}, {len(literal)}")
					for i in range(np.max((len(metaphor),len(contradiction),len(literal)))):
						metaphors.append(metaphor[i].strip())
						contras.append(contradiction[i].strip())
						sources.append(source)
						targets.append(target)
						literal.append(literal[i].strip())
				if len(metaphor)==0:
					print(item)
	except FileNotFoundError:
    		print(f"File not found: {item}")
	except IndexError:
		print(f"Indexing issue with file: {item}")
	except ValueError as ve:
		print(f"Value error: {ve}")
	except Exception as e:
		print(f"Unexpected error for file {item}: {e}")
	except:
		print(f'Dokumentu hau ez da irakurri behar: {item}')
		pass

print(len(sources), len(targets), len(metaphors))
df = pd.DataFrame()
df['source'] = sources
df['target'] = targets
df['metaphor'] = metaphors
df['literal'] = literals
df['contradiction'] = contras

df.to_csv("first_step_extended.csv")
analysis = {}

for index, row in df.iterrows():
	s = row['source']
	t = row['target']
	size = df[(df['source']==s)&(df['target']==t)].shape[0]
	if s+"_"+t+".txt" not in analysis.keys():
		analysis[s+"_"+t+".txt"] = size
		# print(f'Source: {s}, target: {t} and size: {size}')
	else:
		pass

with open('analysis_extended.csv', mode='w', newline='') as file:
	writer = csv.writer(file)
	# Write the header
	writer.writerow(['names', 'values']) 
	# Write the data
	for name, value in analysis.items():
		print(name, value)
		writer.writerow([name, value])
