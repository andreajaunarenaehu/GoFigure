# GoFigure! Seeing Beyond the Literal for the Benchmarking of Vision-Language Models

<p align="center">
   <a href="https://andreajaunarenaehu.github.io/GoFigure_project_page/"> Project Page </a> |
   <a href=""> Paper </a> |
   <a href="https://huggingface.co/datasets/AndreaJaunarena-Cayetano/GoFigure"> GoFigure Dataset </a>
   
</p>

This is the official implementation for the paper GoFigure! Seeing Beyond the Literal for the Benchmarking of Vision-Language Models

## GoFigure dataset

### Dataset Summary

***GoFigure*** is a ***general-purpose multimodal dataset for metaphor understanding***, designed to evaluate how Vision Language Models (VLMs) and Multimodal Large Language Models (MLLMs) process figurative meaning across modalities (images and text). Unlike previous datasets, GoFigure supports ***multiple metaphor-related tasks***, including metaphor ***detection***, ***interpretation***, ***generation***, and ***cross-modal mapping***.

![Examples_web_orria_40_2](https://github.com/user-attachments/assets/6d7a7b31-b45f-482e-bc0b-db7761d54edb)

#### Dataset instances

***GoFigure*** has 639 instances consisting of: 
* ***visual metaphor*** (image), 
* the ***source*** and the ***target*** of the conceptual mapping,
* the ***generated linguistic metaphor***,
* its contradicting metaphor (***contradicting metaphor***),
* the ***entailing literal*** (which expresses the same idea of the linguistic metaphor without using figurative language),
* the ***contradicting literal*** (which expresses an idea that contradicts or is opposite to the original metaphor's meaning),
* a description of the meaning of the generated linguistic metaphor (the concept or idea that the linguistic metaphor conveys in a short sentence) (***literal description***),
* the ***objects***, ***properties***, and ***relations*** that appear in the visual metaphor, and
* the ***visual elaboration*** used to create the visual metaphor.

#### How to use 

To load data with datasets:
```python
>>> ds = load_dataset("AndreaJaunarena-Cayetano/GoFigure")
```

#### Instance example

Each instance has the following structure:
```
{
  'image_path': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1024x1024 at 0x7CFD0EF0D400>,
  'source': 'Getting A Burden',
  'target': 'Intoxication',
  'generated_linguistic_metaphor': 'Her sobriety crumbled under the heavy load of cocktails.',
  'entailing_literal': 'Multiple cocktails gradually eroded her ability to maintain clear-headed behavior.',
  'contradicting_metaphor': 'Each cocktail lifted weight from her shoulders, fortifying her sobriety and self-control.',
  'contradicting_literal': 'The cocktails enhanced her sobriety and self-control.',
  'literal_description': 'Loss of self-control due to excessive drinking.',
  'objects': 'Sobriety, cocktails',
  'properties': 'Crumbling, heavy, structural collapse',
  'relations': 'Weight of cocktails causing destruction of sobriety',
  'visual_elaboration': 'A stone pillar labeled "sobriety" crumbling under a pile of colorful cocktail glasses.'
}
```
## Code Structure

```
.
├── Creation of the dataset                                  #  Folder for creatind the dataset                                
│   ├── Image generation                                     #  Folder for generating the visual metaphors
|        └── dalle-3-generations                             #  Generated visual metaphors
|        └── generation.py                                   #  python code for generation 
|        └── second_step.csv                                 #  metaphor-visual elaboration pairs                   
│   └── Text generation                                      #  Folder for generating the text of the dataset                             
│        └── generations                                     #  Folder for saving the generated text 
│            └── metaphors_generated                         #  Generated metaphors are saved in this folder 
│            └── visual_elaboration                          #  Generated visual elaborations are saved in this folder
│        └── ISM_few_shot_examples.txt                       #  Prompt 
│        └── ISM_modified.txt                                #  Prompt 
│        └── ImageMet_with_contradictions.csv                #  Generated contradicting metaphors with the other text
│        └── final_generation.py                             #  python code for generating the text 
│        └── generate_contradictions.txt                     #  Generated contradicting metaphors
│        └── generate_figurate_contradictions.py             #  python code for generating the contradicting metaphors   
│        └── hirurak_prompt.txt                              #  Prompt    
│        └── our_dataset_updated.csv                         #  Concatenating all text in a csv file  
│        └── source_target_final.csv                         #  Mappings
├── Experiments                                              #  Folder for conducting the experiments 
│   ├── I2T and T2I Retrival                                 #  Folder for running both i2t and t2i retrieval                    
│   │   ├── CLIP                                             #  Experiments with CLIP model 
│            └── clip_retrieval.py                           #  python code for doing retrieval with CLIP 
│            └── results_ImageMet_retrieval_i2t_clip.csv     #  clip i2t results 
│            └── results_ImageMet_retrieval_t2i_clip.csv     #  clip t2i results                         
│   │   └── SigLIP                                           #  Experiments with SigLIP model 
│            └── siglip_retrieval.py                         #  python code for doing retrieval with SigLIP 
│            └── results_ImageMet_retrieval_i2t_siglip.csv   #  siglip i2t results 
│            └── results_ImageMet_retrieval_t2i_siglip.csv   #  siglip t2i results    
│   │   └── SigLIP 2                                         #  Experiments with SigLip 2 model 
│            └── siglip2_retrieval.py                        #  python code for doing retrieval with SigLip 2 
│            └── results_ImageMet_retrieval_i2t_siglip2.csv  #  siglip 2 i2t results 
│            └── results_ImageMet_retrieval_t2i_siglip2.csv  #  siglip 2 t2i results            
│   ├── Metaphor generation                                  #  Folder for running metaphor generation experiments                       
│   │   ├── JudgeLMs rankings                                #  Folder for running judgelms rankings                         
│   │   │   ├── Command A JudgeLM results                    #  results for command a     
│   │   │   └── Prometheus 2 JudgeLM results                 #  results for prometheus 2
│   │   │   └── Qwen 3 JudgeLM results                       #  results for qwen 3 
│   │   │   └── pairwise_ranking_all.py                      #  python code for running the rankings 
│   │   └── Metaphor generation                              #  Folder for running the metaphor generation
│   │       ├── results                                      #  results folder 
│   │       ├── evaluation_few_shot.py                       #  python code for creating the metaphors 
│   ├── Visual Entailment                                    #  Folder for running visual entailment                      
│   │   ├── CLIP                                             #  Experiments with CLIP model 
│   │       ├── clip_ve.py                                   #  python code for running ve with clip 
│   │       ├── results_ImageMet_ve_clip.csv                 #  clip results          
│   │   ├── SigLIP                                           #  Experiments with SigLIP model
│   │       ├── siglip_ve.py                                 #  python code for running ve with siglip 
│   │       ├── results_ImageMet_ve_siglip.csv               #  siglip results 
│   │   ├── SigLIP 2                                         #  Experiments with SigLIP 2 model  
│   │       ├── siglip2_ve.py                                #  python code for running ve with siglip 2 
│   │       ├── results_ImageMet_ve_siglip2.csv              #  siglip 2 results   
│   │   └── Qwen 2.5 VL                                      #  Experiments with Qwen 2.5 VL model  
│   │       ├── qwen2.5_ve.py                                #  python code for running ve with qwen 2.5 VL 
│   │       ├── results_ImageMet_ve_qwen2.5.csv              #  qwen 2.5 VL results 
│   │   └── LLaVA 1.6                                        #  Experiments with LLaVA 1.6 model 
│   │       ├── llava1.6_ve.py                               #  python code for running ve with LLaVA 1.6 
│   │       ├── results_ImageMet_ve_llava1.6.csv             #  llava 1.6 results 
│   │   └── Llama 3.2                                        #  Experiments with Llama 3.2 model 
│   │       ├── llama3.2_ve.py                               #  python code for running ve with llama 3.2
│   │       ├── results_ImageMet_ve_llama3.2.csv             #  llama 3.2 results   
```

## License
See the [LICENSE](./LICENSE) file for details about the license under which code and data is made available.

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation
```
