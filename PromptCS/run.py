# coding=utf-8

from __future__ import absolute_import
import os
import bleu
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from time import time
from model import PromptCS
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def time_format(sec):
    hour = sec//3600
    sec = sec % 3600
    minute = sec//60
    second = sec % 60
    return hour, minute, second


class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename, args):
    """Read examples from filename."""
    examples=[]
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx>50:
                break
            line=line.strip()
            js=json.loads(line)

            source = js['clean_code']
            nl = js['clean_doc']

            examples.append(
                Example(
                        idx=idx,
                        source=source,
                        target=nl,
                        ) 
            )
    return examples


class PromptCSDataset(Dataset):
    def __init__(self, dataset_type, examples, args):
        super().__init__()
        self.args = args
        self.examples = examples
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i].source, self.examples[i].target


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default='../LLMs/codegen-350m', type=str,
                        help="Path to pre-trained model" )
    parser.add_argument("--output_dir", default='./saved_models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default='./saved_models/checkpoint-best-bleu/pytorch_model.bin', type=str,
                        help="Path to trained model: Should contain the .bin files" )
    parser.add_argument("--early_stop", default=4, type=int,
                        help="Patience for early stop.")
    parser.add_argument("--reload", default=False, type=bool,
                        help="Whether to reload the previous model")
    parser.add_argument("--mode", default='PromptCS', type=str,
                        choices=["PromptCS", "finetune"],
                        help="Operational mode.")
    parser.add_argument("--template", type=str, default="[0, 100]",
                        help="The concatenation method of pseudo tokens and code snippet.")
    parser.add_argument("--prompt_encoder_type", default='lstm', type=str,
                        choices=["lstm", "transformer"],
                        help="Architecture of prompt encoder.")
    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--train_data_size", default=-1, type=int,
                        help="Dataset to be used during training. If -1, take the entire train dataset. Otherwise, take subset.")
    parser.add_argument("--num_train_epochs", default=50, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--train_filename", default='../dataset/java/clean_train.jsonl', type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default='../dataset/java/clean_valid.jsonl', type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default='../dataset/java/clean_test.jsonl', type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")  

    parser.add_argument("--max_code_length", default=300, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=30, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", default=True, type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, type=bool,
                        help="Whether to run eval on the dev set during training.")
    parser.add_argument("--do_test", default=True, type=bool,
                        help="Whether to run testing on the test dataset.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--seed', type=int, default=2023,
                        help="random seed for initialization")

    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, "args.info"), 'w', encoding='utf-8') as f:
        f.write(args.__str__())

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    assert args.n_gpu == 1, ("This version of PromptCS can only run on a single GPU \n"
                             "Please set the GPU you want to use and run again. For example: CUDA_VISIBLE_DEVICES=0 python run.py \n"
                             "If you need multi-GPU training, please check out the DeepSpeed version of PromptCS")

    logger.warning("model: %s, prompt_encoder: %s, len: %s, dataset: %s",
                   args.model_name_or_path, args.prompt_encoder_type, args.template, args.train_filename)

    args.device = device
    args.template = eval(args.template)
    set_seed(args.seed)

    #Build model
    model=PromptCS(args=args, device=device, template=args.template)

    if args.reload:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
    model.to(device)

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename, args)
        if args.train_data_size != -1:
            train_examples = random.sample(train_examples,min(args.train_data_size, len(train_examples)))

        train_data = PromptCSDataset('train', train_examples, args)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # Prepare optimizer and schedule (linear warmup and decay)
        t_total = len(train_dataloader) * args.num_train_epochs
        optimizer = AdamW(filter(lambda p: p.requires_grad,
                                 model.parameters()), lr=args.learning_rate, weight_decay=0.0001, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total*0.1),
                                                    num_training_steps=t_total)
    
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)

        model.train()
        dev_dataset={}
        nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0,0,0,0,1e6
        early_stopping_flag = 0
        start = time()
        checkpoint_start_time = time()
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader,total=len(train_dataloader), ncols=150)
            for batch in bar:
                codes, docs = batch
                loss = model(codes, docs)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
                bar.set_description("epoch {} loss {}".format(epoch,train_loss))
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            if args.do_eval:
                checkpoint_end_time = time()
                early_stopping_flag += 1
                #Calculate bleu  
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename, args)
                    eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                    eval_data = PromptCSDataset('dev', eval_examples, args)
                    dev_dataset['dev_bleu'] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                p=[]
                all_out = []
                bar = tqdm(eval_dataloader, total=len(eval_dataloader), ncols=150)
                for batch in bar:
                    with torch.no_grad():
                        codes, docs = batch
                        preds = model(x_hs=codes)
                        for pred in preds:
                            all_out.append(pred)
                            if '.' in pred:
                                pred = pred[:pred.index('.')]+' .'
                            p.append(pred)

                model.train()

                predictions=[]
                with open(os.path.join(args.output_dir,"dev.output"),'w', encoding='utf-8') as f, open(os.path.join(args.output_dir,"dev.gold"),'w',encoding='utf-8') as f1, open(os.path.join(args.output_dir, "dev.all.output"), 'w', encoding='utf-8') as f2:
                    for ref,gold,raw in zip(p,eval_examples, all_out):
                        predictions.append(str(gold.idx)+'\t'+ref)
                        f.write(str(gold.idx)+'\t'+ref+'\n')
                        f1.write(str(gold.idx)+'\t'+gold.target+'\n')
                        f2.write(str(gold.idx)+'\t'+raw+'\n')

                (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold")) 
                dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
                logger.info("  " + "*" * 20)
                logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                logger.info("  "+"*"*20)

                hour, minute, second = time_format(checkpoint_end_time - checkpoint_start_time)
                with open(os.path.join(args.output_dir, "time.info"), 'a', encoding='utf-8') as f:
                    f.write("  Training Time of the epoch: {} h {} m {} s, eval bleu: {} \n".format(hour, minute, second, dev_bleu))

                if dev_bleu>best_bleu:
                    early_stopping_flag = 0
                    logger.info("  Best bleu:%s",dev_bleu)
                    logger.info("  "+"*"*20)

                    best_bleu=dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

                checkpoint_start_time = time()

            if early_stopping_flag >= args.early_stop:
                break
        end = time()
        hour, minute, second = time_format(end - start)
        logger.info("  Training time: %d h %d m %d s", hour, minute, second)
        logger.info("***** End training *****")

        with open(os.path.join(args.output_dir, "time.info"), 'a', encoding='utf-8') as f:
            f.write("  Total Training time: {} h {} m {} s \n".format(hour, minute, second))


    if args.do_test:
        model = PromptCS(args=args, device=device, template=args.template)

        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

        model.to(device)

        files=[]
        if args.test_filename is not None:
            files.append(args.test_filename)
        for idx,file in enumerate(files):   
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file, args)
            eval_data = PromptCSDataset('test', eval_examples, args)

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            start = time()
            p=[]
            all_out = []
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader), ncols=150):
                with torch.no_grad():
                    codes, docs = batch
                    preds = model(x_hs=codes)
                    for pred in preds:
                        all_out.append(pred)
                        if '.' in pred:
                            pred = pred[:pred.index('.')]+' .'
                        p.append(pred)

            model.train()
            predictions=[]
            with open(os.path.join(args.output_dir,"test_{}.output".format(str(idx))),'w',encoding='utf-8') as f, open(os.path.join(args.output_dir,"test_{}.gold".format(str(idx))),'w',encoding='utf-8') as f1, open(os.path.join(args.output_dir, "test.all.output"), 'w', encoding='utf-8') as f2:
                for ref,gold,raw in zip(p,eval_examples,all_out):
                    predictions.append(str(gold.idx)+'\t'+ref)
                    f.write(str(gold.idx)+'\t'+ref+'\n')
                    f1.write(str(gold.idx)+'\t'+gold.target+'\n')
                    f2.write(str(gold.idx) + '\t' + raw + '\n')

            (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "test_{}.gold".format(idx))) 
            dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
            logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
            logger.info("  "+"*"*20)

            end = time()
            hour, minute, second = time_format(end - start)
            logger.info("  Testing time: %d h %d m %d s", hour, minute, second)
            logger.info("***** End testing *****")

            with open(os.path.join(args.output_dir, "time.info"), 'a', encoding='utf-8') as f:
                f.write("  Total Testing time: {} h {} m {} s, bleu: {} \n".format(hour, minute, second, dev_bleu))


if __name__ == "__main__":
    main()


