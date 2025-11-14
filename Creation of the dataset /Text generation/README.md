This README file helps to understand the files that are in this folder:

- source_target_final.csv: This file contains all the mappings that we use (source domain and target domain).
- final_generation.py: This is the code to generate the entailing literals/metaphors, the contradicting literals and the visual elaboration. Inside the code you have instructions to generate on the one hand the entailing literals/metaphors and the contradicting literals, and on the other hand the visual elaborations. 
- generations: This is a folder where you can find two subfolders with the generated data. In the metaphors_generated folder, you can find the entailing literals/metaphors and the contradicting literals. In the visual_elaboration folder, you can find the visual elaborations for each entailing metaphor. 
- hirurak_prompt.txt: The prompt used for creating the entailing literals/metaphors and the contradicting literals. 
- ISM_modified.text: The prompt used for creating the visual elaborations. This prompt has been taken from HAIVMet dataset paper (https://arxiv.org/abs/2305.14724).
- ISM_modified.txt: The prompt for creating the visual elaborations contains 5 examples that are given in this file. The examples are also taken from the HAIVMet dataset paper (https://arxiv.org/abs/2305.14724). 
- our_dataset_updated.csv: ImageMet dataset without contradicting metaphors 
- generate_contradictions.txt: Prompt to generate contradicting metaphors
- ImageMet_with_contradictions.csv: ImageMet dataset with both entailing/contradicting metaphors and literals. Additional information such as which is the figurative meaning of the contradicting metaphor can be found. 