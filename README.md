# PromptCS
A Prompt Learning Framework for Source Code Summarization

## Dependency
* python==3.8
* torch==2.1.0
* transformers==4.32.1
* deepspeed==0.12.2 (option)
* openai==0.28.0 (option)


## Dataset
We use the Java, Javascript and Python dataset from the [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text) code-to-text docstring
generation task, which is built upon the CodeSearchNet corpus and
excludes defective data samples. 

### Download data and preprocess

    unzip dataset.zip
    cd dataset
    wget https://zenodo.org/record/7857872/files/java.zip
    wget https://zenodo.org/record/7857872/files/javascript.zip
    wget https://zenodo.org/record/7857872/files/python.zip
    
    unzip python.zip
    unzip java.zip
    unzip javascript.zip

    python preprocess.py

    rm *.pkl
    rm -r */[^clean]*
    cd ..


### Data Format

After preprocessing dataset, you can obtain three .jsonl files, i.e. clean_train.jsonl, clean_valid.jsonl, clean_test.jsonl

For each file, each line in the uncompressed file represents one function. Here is an explanation of the fields:

* The fields contained in the original CodeXGLUE dataset:

  * repo: the owner/repo

  * path: the full path to the original file

  * func_name: the function or method name

  * original_string: the raw string before tokenization or parsing

  * language: the programming language

  * code/function: the part of the original_string that is code

  * code_tokens/function_tokens: tokenized version of code

  * docstring: the top-level comment or docstring, if it exists in the original string

  * docstring_tokens: tokenized version of docstring

* The additional fields we added:

  * clean_code: clean version of code that removing possible comments

  * clean_doc: clean version of docstring that obtaining by concatenating docstring_tokens

### Data Statistic

| Programming Language | Training |  Dev   |  Test  |
| :------------------- | :------: | :----: | :----: |
| Python               | 251,820  | 13,914 | 14,918 |
| Java                 | 164,923  | 5,183  | 10,955 |
| JavaScript           |  58,025  | 3,885  | 3,291  |


## Training of PromptCS

### Quick Start
    cd PromptCS
    CUDA_VISIBLE_DEVICES=0 python run.py --mode PromptCS --prompt_encoder_type lstm --template [0,100] --model_name_or_path ../LLMs/codegen-350m --train_filename ../dataset/java/clean_train.jsonl --dev_filename ../dataset/java/clean_valid.jsonl --test_filename ../dataset/java/clean_test.jsonl --output_dir ./saved_models --train_batch_size 16 --eval_batch_size 16 --learning_rate 5e-5 

On a single A100 or A800, our experimental results can be reproduced in this way.
However, it can only run on a single GPU.
If your device has insufficient GPU memory, or you need multi-GPU training, please check out the DeepSpeed version of training PromptCS

### Train with DeepSpeed
We set the Zero Redundancy Optimizer (ZeRO) to ZeRO-3 and enable the offloading of optimizer computation to CPU.

However, the experimental results that can be obtained using deepspeed to train promptCS or finetune LLMs is unvalidated.

    cd PromptCS-DeepSpeed
    deepspeed --num_gpus=2 run.py

### Arguments
The explanation for some of the arguments is as follows.

* model_name_or_path: Path to pre-trained model
* mode: Operational mode. Choices=["PromptCS", "finetune"]
* prompt_encoder_type: Architecture of prompt encoder. Choices=["lstm", "transformer"]
* template: The concatenation method of pseudo tokens and code snippet. Default is the Back-end Mode [0, 100]
* output_dir: The output directory where the model predictions and checkpoints will be written.

For a complete list of all arguments settings, please refer to the 'run.py'.

## Evaluation

### BLEU and SentenceBERT
    cd PromptCS
    python evaluate.py --predict_file_path ./saved_models/test_0.output --ground_truth_file_path ./saved_models/test_0.gold --SentenceBERT_model_path ../all-MiniLM-L6-v2

### METEOR and ROUGE-L
To obtain METEOR and ROUGE-L, we need to activate the environment that contains python 2.7

    conda activate py27
    unzip evaluation
    cd evaluation
    python evaluate.py --predict_file_path ../PromptCS/saved_models/test_0.output --ground_truth_file_path ../PromptCS/saved_models/test_0.gold

Tip: The path should only contain English characters.

## Zero-Shot LLMs
    cd zeroshot
    python manual.py --model_name_or_path ../bigcode/starcoderbase-3b --test_filename ../dataset/java/clean_test.jsonl
    python manual_gpt_3.5.py --test_filename ../dataset/java/clean_test.jsonl

## Few-Shot LLMs
We directly leverage the 10 Java examples provided by Ahmed et al. in their GitHub [repository](https://github.com/toufiqueparag/few_shot_code_summarization/tree/main/Java), since we use the same experimental dataset (i.e., the CSN corpus).

    cd fewshot
    python manual.py --model_name_or_path ../bigcode/starcoderbase-3b --test_filename ../dataset/java/clean_test.jsonl
    python manual_gpt_3.5.py --test_filename ../dataset/java/clean_test.jsonl