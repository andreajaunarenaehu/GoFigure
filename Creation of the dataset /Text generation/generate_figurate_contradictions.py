import os
import re
import pandas as pd
import anthropic
import time

API_KEY = "" # write here you anthropic key 
client = anthropic.Anthropic(
    api_key=API_KEY,
)

# ImageMet dataset without contradicting metaphors 
ImageMet = pd.read_csv("our_dataset_updated.csv")

# Prompt to generate contradicting metaphors
with open("generate_contradictions.txt", "r") as file: 
    prompt = file.read()

print(f'Prompt to generate contradictory metaphors: {prompt}')

# File where generations will be loaded 
output_path = "ImageMet_with_contradictions.csv"

if os.path.exists(output_path):
    result = pd.read_csv(output_path)
    print(f"Loaded existing results with {len(result)} rows.")
else:
    result = pd.DataFrame(columns=[
        "source",
        "target",
        "literal metaphor",
        "contradiction metaphor",
        "figurative meaning of the contradiction metaphor",
        "literal meaning",
        "contradiction meaning"
    ])

processed_metaphors = set(result["literal metaphor"].dropna())

result = pd.DataFrame()

# Code for generating and loading the contradicting metaphors.
for index, row in ImageMet.iterrows():
    metaphor = row['metaphor']
    if metaphor in processed_metaphors:
        print(f"Skipping already processed metaphor: {metaphor}")
        continue

    source = row['source']
    target = row['target']
    literal = row['literal']
    contradiction = row['contradiction']
    print(f'Source: {source}')
    print(f'Target: {target}')
    print(f'Metaphor: {metaphor}')
    print(f'Literal meaning: {literal}')
    print(f'Contradictory meaning: {contradiction}')
    prompt2 = prompt.replace("{{SOURCE}}", source).replace("{{TARGET}}", target).replace("{{ORIGINAL_METAPHOR}}", metaphor).replace("{{LITERAL_PARAPHRASE}}", literal).replace("{{CONTRADICTION_PARAPHRASE}}", contradiction)
    message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=20000,
    temperature=1,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<examples>\n<example>\n<SOURCE>\nTime\n</SOURCE>\n<TARGET>\nMoney\n</TARGET>\n<ORIGINAL_METAPHOR>\nTime is a valuable currency.\n</ORIGINAL_METAPHOR>\n<LITERAL_PARAPHRASE>\nTime is a limited and important resource that should be used wisely.\n</LITERAL_PARAPHRASE>\n<CONTRADICTION_PARAPHRASE>\nTime is an infinite and worthless commodity that can be wasted freely.\n</CONTRADICTION_PARAPHRASE>\n<ideal_output>\nTime is monopoly money from an infinite bank, freely squandered without consequence.\n</ideal_output>\n</example>\n</examples>\n\n"
                },
                {
                    "type": "text",
                    "text": prompt2
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "<analysis>"
                }
            ]
        }
    ]
)
    text = message.content[0].text
    contradictory_metaphor_match = re.search(r"<contradictory_metaphor>(.*?)</contradictory_metaphor>", text, re.DOTALL)
    contradictory_metaphor = contradictory_metaphor_match.group(1).strip() if contradictory_metaphor_match else None

    figurative_language_identification_match = re.search(r"<figurative_language_identification>(.*?)</figurative_language_identification>", text, re.DOTALL)
    figurative_language_identification = figurative_language_identification_match.group(1).strip() if figurative_language_identification_match else None

    print(f"Contradictory metaphor: {contradictory_metaphor}")
    print(f"Figurative language indetification: {figurative_language_identification}")
    print()

    new_row = pd.DataFrame([{
        "source": source,
        "target": target,
        "literal metaphor": metaphor,
        "contradiction metaphor": contradictory_metaphor,
        "figurative meaning of the contradiction metaphor": figurative_language_identification,
        "literal meaning": literal,
        "contradiction meaning": contradiction,
    }])

    result = pd.concat([result, new_row], ignore_index=True)
    result.to_csv(output_path, index=False)
    print(f"Saved result for: {metaphor}")
    time.sleep(1)

