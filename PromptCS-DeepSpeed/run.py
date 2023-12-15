# coding=utf-8

from __future__ import absolute_import
import os
import bleu
import deepspeed
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
from torch.utils.data.distributed import DistributedSampler

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


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


def read_examples(filename, mode):
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

def master_process(args):
    return args.local_rank == -1 or torch.distributed.get_rank() == 0

def master_logger(msg, args):
    if master_process(args):
        logger.info(msg)

def barrier(args):
    if args.local_rank != -1:
        deepspeed.comm.barrier()

def deepspeed_init(model, args):
    print("Creating DeepSpeed engine...")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps":  1,
        "steps_per_print": 2000,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": args.learning_rate,
                "betas": (0.9, 0.95),
                "eps": args.adam_epsilon,
                "weight_decay": args.weight_decay,
                "adam_w_mode": True,
                "torch_adam": True
            }
        },
        "gradient_clipping": 0.5,
        "prescale_gradients": False,
        "fp16": {
            "enabled": False,
            "auto_cast": True,
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 50,
            "hysteresis": 2,
            "min_loss_scale": 32.0,
            "initial_scale_power": 13
        },
        "amp": {
            "enabled": False,
            "opt_level": "O1",
            "min_loss_scale": 32.0,
        },
        "bfloat16": {
            "enabled": False
        },
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
            },
            "contiguous_gradients": True,
            "overlap_comm": True,
        },
        "flops_profiler": {
            "enabled": False,
            "profile_step": 10,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
            "output_file": None,
            "end_profile_step": 5,
        },
    }

    model_engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)

    return model_engine

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name_or_path", default='../bigcode/starcoderbase-7b', type=str,
                        help="Path to pre-trained model" ) 
    parser.add_argument("--output_dir", default='/home/yyd/saved_model/starcoderbase-7b-finetune', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--valid_every", default=1, type=int,
                        help="Number of training epochs between twice validation." )
    parser.add_argument("--early_stop_patience", default=4, type=int,
                        help="Patience for early stop.")
    parser.add_argument("--reload", default=False, type=bool,
                        help="Whether to reload the previous model")
    parser.add_argument("--mode", default='finetune', type=str,
                        choices=["PromptCS", "finetune"],
                        help="Operational mode.")
    parser.add_argument("--template", type=str, default="[0, 100]",
                        help="The concatenation method of pseudo tokens and code snippet.")

    parser.add_argument("--batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU.")
    parser.add_argument("--num_train_epochs", default=50, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")

    ## Other parameters
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
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=True, type=bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=2023,
                        help="random seed for initialization")

    # print arguments
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        deepspeed.init_distributed(dist_backend='nccl')

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    args.template = eval(args.template)
    set_seed(args.seed)

    if master_process(args):
        logger.info(args)
        if os.path.exists(args.output_dir) is False:
            os.makedirs(args.output_dir)

        with open(os.path.join(args.output_dir, "args.info"), 'w', encoding='utf-8') as f:
            f.write(args.__str__())

    barrier(args)
    #build model
    net = PromptCS(args=args, device=device, template=args.template)
    net.to(device)

    model_engine = deepspeed_init(net, args)
    if args.reload:
        logger.info("reload model from {}".format(args.output_dir))
        _,_ = model_engine.load_checkpoint(args.output_dir, "best_bleu_model")

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename, args.mode)
        train_data = PromptCSDataset('train', train_examples, args)
        
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

        master_logger("***** Running training *****", args)
        master_logger("  Num examples = {}".format(len(train_examples)), args)
        master_logger("  Batch size per GPU = {}".format(args.batch_size), args)
        master_logger("  Num epoch = {}".format(args.num_train_epochs), args)

        dev_dataset={}
        nb_tr_steps,global_step,best_bleu,best_loss = 0,0,0,1e6
        early_stopping_flag = 0
        start = time()
        checkpoint_start_time = time()
        for epoch in range(args.num_train_epochs):
            model_engine.train()
            if master_process(args):
                train_dataloader = tqdm(train_dataloader, total=len(train_dataloader), ncols=150)
            for batch in train_dataloader:
                codes, docs = batch
                loss = model_engine(codes, docs)

                model_engine.backward(loss)
                model_engine.step()
                nb_tr_steps += 1

            if args.do_eval:
                checkpoint_end_time = time()
                early_stopping_flag += 1

                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename, args.mode)
                    eval_examples = random.sample(eval_examples,min(100,len(eval_examples)))
                    eval_data = PromptCSDataset('dev', eval_examples, args)
                    dev_dataset['dev_bleu'] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

                master_logger("***** Running evaluation *****", args)
                master_logger("  Num examples = {}".format(len(eval_examples)), args)
                master_logger("  Batch size = {}".format(args.batch_size), args)

                model_engine.eval()
                p=[]
                if master_process(args):
                    eval_dataloader = tqdm(eval_dataloader, total=len(eval_dataloader), ncols=150)

                for batch in eval_dataloader:
                    with torch.no_grad():
                        codes, docs = batch
                        preds = model_engine(x_hs=codes)
                        for pred in preds:
                            if '.' in pred:
                                pred = pred[:pred.index('.')]+' .'
                            p.append(pred)

                predictions = []

                if master_process(args):
                    with open(os.path.join(args.output_dir, "dev.output"),'w', encoding='utf-8') as f, open(os.path.join(args.output_dir,"dev.gold"),'w',encoding='utf-8') as f1:
                        for ref,gold in zip(p,eval_examples):
                            predictions.append(str(gold.idx)+'\t'+ref)
                            f.write(str(gold.idx)+'\t'+ref+'\n')
                            f1.write(str(gold.idx)+'\t'+gold.target+'\n')

                    (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold"))
                    dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)


                    logger.info("  " + "*" * 20)
                    logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                    logger.info("  "+"*"*20)

                    hour, minute, second = time_format(checkpoint_end_time - checkpoint_start_time)
                    with open(os.path.join(args.output_dir, "time.info"), 'a', encoding='utf-8') as f:
                        f.write("  Training Time: {} h {} m {} s, eval bleu: {} \n".format(hour, minute, second, dev_bleu))

                barrier(args)
                if not master_process(args):
                    for ref, gold in zip(p, eval_examples):
                        predictions.append(str(gold.idx) + '\t' + ref)
                    (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold"))
                    dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)

                barrier(args)
                try:
                    if dev_bleu>best_bleu:
                        early_stopping_flag = 0
                        master_logger("  Best bleu:{}".format(dev_bleu), args)
                        master_logger("  "+"*"*20, args)

                        best_bleu=dev_bleu
                        # Save best checkpoint for best bleu
                        if not os.path.exists(args.output_dir):
                            os.makedirs(args.output_dir)
                        model_engine.save_checkpoint(args.output_dir, "best_bleu_model")
                except:
                    logger.info("Save model fail")

                checkpoint_start_time = time()

            if early_stopping_flag >= args.early_stop_patience:
                break
        end = time()
        hour, minute, second = time_format(end - start)
        master_logger("  Training time: {} h {} m {} s".format(str(hour), str(minute), str(second)), args)
        master_logger("***** End training *****", args)

        if master_process(args):
            with open(os.path.join(args.output_dir, "time.info"), 'a', encoding='utf-8') as f:
                f.write("  Total Training time: {} h {} m {} s \n".format(hour, minute, second))


    if args.do_test:
        master_logger("device {} reload model from {}".format(device, args.output_dir), args)
        _, _ = model_engine.load_checkpoint(args.output_dir, "best_bleu_model")

        files=[]
        if args.test_filename is not None:
            files.append(args.test_filename)
        for idx,file in enumerate(files):
            eval_examples = read_examples(file, args.mode)
            eval_data = PromptCSDataset('test', eval_examples, args)

            master_logger("***** Running testing *****", args)
            master_logger("  Num examples = {}".format(len(eval_examples)), args)
            master_logger("Test file: {}".format(file), args)

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

            start = time()
            p=[]
            model_engine.eval()
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader), ncols=150):
                with torch.no_grad():
                    codes, docs = batch
                    preds = model_engine(x_hs=codes)
                    for pred in preds:
                        if '.' in pred:
                            pred = pred[:pred.index('.')]+' .'
                        p.append(pred)

            if master_process(args):
                predictions=[]
                with open(os.path.join(args.output_dir,"test_{}.output".format(str(idx))),'w',encoding='utf-8') as f, open(os.path.join(args.output_dir,"test_{}.gold".format(str(idx))),'w',encoding='utf-8') as f1:
                    for ref,gold in zip(p,eval_examples):
                        predictions.append(str(gold.idx)+'\t'+ref)
                        f.write(str(gold.idx)+'\t'+ref+'\n')
                        f1.write(str(gold.idx)+'\t'+gold.target+'\n')

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

            deepspeed.comm.barrier()



if __name__ == "__main__":
    main()


