import re
import os
import pandas as pd

directory_path = '.'

items = os.listdir(directory_path)

metaphors = []
linguistics = []
implicits = []
objects = []
properties = []
relations = []
visuals = []
sources = []
targets = []

metaphor_pattern = r"Metaphor: (.*?)<metaphor_analysis>"
linguistic_pattern = r"<linguistic_metaphor>(.*?)</linguistic_metaphor>"
implicit_pattern = r"<implicit_meaning>(.*?)</implicit_meaning>"
object_pattern = r"<object>(.*?)</object>"
properties_pattern = r"<properties>(.*?)</properties>"
relations_pattern = r"<relations>(.*?)</relations>"
visual_elaboration_pattern = r"<visual_elaboration>(.*?)</visual_elaboration>"

df = pd.read_csv("../metaphors_generated/first_step.csv")

for item in items:
	print(f'Item: {item}')
	if item!= "read.py":
		with open(item,'r') as file:
			lines = " ".join(file.readlines())
			metaphor =  item.split(".txt")[0].strip()
			source = df.iloc[int(metaphor)]['source']
			target = df.iloc[int(metaphor)]['target']
			linguistic = re.findall(linguistic_pattern, lines, re.DOTALL)[0].strip()
			implicit = re.findall(implicit_pattern, lines, re.DOTALL)[0].strip()
			object = re.findall(object_pattern, lines, re.DOTALL)[0].strip()
			prop = re.findall(properties_pattern, lines, re.DOTALL)[0].strip()
			rel = re.findall(relations_pattern, lines, re.DOTALL)[0].strip()
			vis = re.findall(visual_elaboration_pattern, lines, re.DOTALL)[0].strip()
			if linguistic not in linguistics:
				metaphors.append(metaphor)
				linguistics.append(linguistic)
				implicits.append(implicit)
				objects.append(object)
				properties.append(prop)
				relations.append(rel)
				visuals.append(vis)
				sources.append(source)
				targets.append(target)

df = pd.DataFrame()
df['source'] = sources
df['target'] = targets
df['metaphor'] = metaphors
df['linguistic_metaphor'] = linguistics
df['implicit_meaning'] = implicits
df['object'] = objects
df['properties'] = properties
df['relations'] = relations
df['visual_elaboration'] = visuals

df.to_csv("second_step.csv")


