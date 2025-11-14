import pandas as pd
import openai
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
import os

API_KEY = "" # write your key here
client = OpenAI(
    api_key=API_KEY,
)

# Csv containing entailing meatphors and visual elaborations to generate the visual metaphors
df = pd.read_csv("second_step.csv")
path_dir = 'dalle-3-generations'

files_and_dirs = os.listdir(path_dir)

# Filter out only files
files = [f for f in files_and_dirs if os.path.isfile(os.path.join(path_dir, f))]

print("Files:", files)

# Code for generating 4 visual metaphors per visual elaboration
for index, row in df.iterrows():
	print(index)
	p = row['visual_elaboration']
	i = row['metaphor']
	print(f'Visual elaboration: {p}')
	names = [f'{i}_0.jpg', f'{i}_1.jpg', f'{i}_2.jpg', f'{i}_3.jpg']
	zenbat = []
	for n_index, name in enumerate(names):
		if name in files:
			print(f'I have already generate images for {p}')
			continue
		else:
			zenbat.append(n_index)
	for z in zenbat:
		# print(z)
		try:
			print(z)
			response = client.images.generate(
				model = "dall-e-3",
				prompt = p,
				size = "1024x1024",
				quality = "standard",
				n = 1
			)
			image_url = response.data[0].url
			# print(image_url)
			response = requests.get(image_url)
			# print(response)
			if response.status_code == 200:
				image = Image.open(BytesIO(response.content))
				image.save(f'{path_dir}/{i}_{z}.jpg')
				print("Image saved successfully!")
			else:
				print(f'Failed to retrieve image. HTTP Status Code: {response.status_code}')
		except openai.OpenAIError as e:
			print(e)
			print(f'Visual elaboration error: {p}')