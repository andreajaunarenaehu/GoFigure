import csv
import re
import os
import pandas as pd
from collections import Counter

directory_path = '/scratch/ajaunarena/claude-env/final_prompt/metaphors_generated/'

items = os.listdir(directory_path)
print(len(items))
df = pd.read_csv("analysis.csv") 
a = df['names'].to_list()
print(len(a))
b = pd.read_csv("../../source_target_final.csv")
lista = []
for index, row in b.iterrows():
	lista.append(row['source']+"_"+row['target']+".txt")
for i in items:
	if i not in lista:
		print(i)
