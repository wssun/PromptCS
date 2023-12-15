from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse
import torch
from tqdm import tqdm
import logging
import os
import bleu
import re


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename, few_shot_tensor, nl_begin_tensor, max_code_length):
    """Read examples from filename."""
    examples=[]
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)

            code = js['clean_code']
            nl = js['clean_doc']

            source_tensor = tokenizer.encode(code, return_tensors="pt")[:, :max_code_length]
            source_tensor = torch.cat((few_shot_tensor, source_tensor), dim=1)
            source_tensor = torch.cat((source_tensor, nl_begin_tensor), dim=1)

            examples.append(
                Example(
                        idx=idx,
                        source=source_tensor,
                        target=nl
                        )
            )
    return examples


def prepare_prompt(model_name_or_path):
    with open("train_10.txt", "r", encoding="utf-8") as f:
        few_shot_sentence = ""
        nl_begin_token = '<s>'
        lines = f.readlines()
        if 'polycoder' in model_name_or_path or 'codegen' in model_name_or_path:
            # Considering the length limitations imposed by CodeGen and PolyCoder, we can only use 7 examples for few-shot them.
            lines = [line.replace('<s>', '// <s>') for line in lines[3:]]
            nl_begin_token = '// <s>'

        for line in lines:
            few_shot_sentence += line

    few_shot_tensor = tokenizer.encode(few_shot_sentence, return_tensors="pt")
    nl_begin_tensor = tokenizer.encode(nl_begin_token, return_tensors="pt")

    return few_shot_tensor, nl_begin_tensor


def get_pad_token_id(model_name, tokenizer):
    pad_token_id = None
    model_name = model_name.lower()
    if 'starcoder' in model_name:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.unk_token_id
    elif 'polycoder' in model_name:
        pad_token_id = tokenizer.get_vocab()['<|padding|>']
    elif 'codegen' in model_name:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.unk_token_id

    return pad_token_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name_or_path", default='../bigcode/starcoderbase-3b', type=str,
                        help="Path to pre-trained model")
    parser.add_argument("--output_dir", default='./saved_fewshot_result', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--test_filename", default='../dataset/java/clean_test.jsonl', type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--max_code_length", default=300, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    # print arguments
    args = parser.parse_args()
    logger.info(args)

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, "args.info"), 'w', encoding='utf-8') as f:
        f.write(args.__str__())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)

    pad_token_id = get_pad_token_id(args.model_name_or_path, tokenizer)

    few_shot_tensor, nl_begin_tensor= prepare_prompt(args.model_name_or_path.lower())

    examples = read_examples(args.test_filename, few_shot_tensor, nl_begin_tensor, args.max_code_length)
    bar = tqdm(examples, total=len(examples), ncols=150)

    p = []
    all_out = []
    for item in bar:
        inputs = item.source.to(device)
        answer_begin_idx = inputs.shape[1]
        inputs = model.generate(inputs, max_new_tokens=30, pad_token_id=pad_token_id)
        answer = inputs[0][answer_begin_idx:]
        answer = tokenizer.decode(answer)
        answer = answer.replace('\n', '').replace('\t', ' ')
        answer = re.sub(r"\s+", " ", answer)
        all_out.append(answer)
        if '</s>' in answer:
            answer = answer[:answer.index('</s>')]
        if '.' in answer:
            answer = answer[:answer.index('.')] + ' .'

        p.append(answer)

    predictions = []
    with open(os.path.join(args.output_dir, "test_0.output"), 'w', encoding='utf-8') as f, open(
            os.path.join(args.output_dir, "test_0.gold"), 'w', encoding='utf-8') as f1, open(
            os.path.join(args.output_dir, "test_0.all.output"), 'w', encoding='utf-8') as f2:
        for ref, gold, raw in zip(p, examples, all_out):
            predictions.append(str(gold.idx) + '\t' + ref)
            f.write(str(gold.idx) + '\t' + ref + '\n')
            f1.write(str(gold.idx) + '\t' + gold.target + '\n')
            f2.write(str(gold.idx) + '\t' + raw + '\n')

    (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "test_0.gold"))
    test_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    logger.info("  " + "*" * 20)
    logger.info("  %s = %s " % ("bleu-4", str(test_bleu)))
    logger.info("  " + "*" * 20)

    with open(os.path.join(args.output_dir, "bleu.info"), 'a', encoding='utf-8') as f:
        f.write("  Bleu {}\n".format(test_bleu))

