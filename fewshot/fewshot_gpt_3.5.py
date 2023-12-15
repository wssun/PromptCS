import openai
import os
import json
import time
import logging
import argparse
import random

logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

openai.api_key = 'your api key'


class Example(object):
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename):
    with open("train_10.txt", "r", encoding="utf-8") as f:
        few_shot_sentence = f.read()

    examples=[]
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)

            code = js['clean_code']
            nl = js['clean_doc']

            query = few_shot_sentence + code + '\t<s>'

            examples.append(
                Example(
                        idx=idx,
                        source=query,
                        target=nl
                        )
            )
    return examples


def set_seed(seed=2023):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default='./saved_fewshot_gpt_3.5_result', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--test_filename", default='../dataset/java/clean_test.jsonl', type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")

    # print arguments
    args = parser.parse_args()

    set_seed()

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    fh = logging.FileHandler(f'{args.output_dir}/log.txt')
    logger.addHandler(fh)  # add the handlers to the logger
    logger.info(args)

    data = read_examples(args.test_filename)
    data = random.sample(data, min(200, len(data)))

    logger.info("Test data size %s", str(len(data)))

    f_output = open(os.path.join(args.output_dir, "test_0.output"), 'w', encoding='utf-8')
    f_gold = open(os.path.join(args.output_dir, "test_0.gold"), 'w', encoding='utf-8')
    flag=True
    idx = 0  # episode!!!
    while idx < len(data):
        try:
            # if idx==2 and flag:
            #     flag=False
            #     a=1/0

            item = data[idx]
            query = item.source
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query},
                ],
            )
            reply = chat['choices'][0]['message']['content']

            if '</s>' in reply:
                reply = reply[:reply.index('</s>')]

            if '.' in reply:
                reply = reply[:reply.index('.')] + ' .'

            logger.info("%s\t%s", str(idx), reply)
            f_output.write(str(idx) + '\t' + reply + '\n')
            f_gold.write(str(idx) + '\t' + item.target + '\n')
            idx += 1
        except:
            time.sleep(30)

    f_gold.close()
    f_output.close()
