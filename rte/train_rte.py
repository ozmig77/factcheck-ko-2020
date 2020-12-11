import os
import json
import random
import argparse
import logging
import socket

import pandas as pd
import numpy as np
#import kss
from tqdm import tqdm, trange
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import (
    TensorDataset, 
    DataLoader, 
    RandomSampler,
    SequentialSampler,
)
from torch.nn.utils import clip_grad_norm_

from transformers import BertTokenizer, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule


logger = logging.getLogger(__name__)



def calculate_metric(logits, labels):
    pred = np.argmax(logits, axis=1).flatten()
    gold = labels.flatten()
    return np.sum(pred == gold) / len(gold)

def get_dataset(dir, split='train'):
    with open('data/{}_anno.json'.format(split), 'r') as f:
        jsondata = json.load(f)
    dataset = []
    name2label = {'TRUE':0, 'FALSE':1, 'NEI':2}
    for elem in jsondata:
        #grammar = elem['grammar']
        #if int(grammar) <3:
        #    continue
        reference = elem['reference']
        query = elem['paraphrased']
        label = name2label[elem['True_False']]
        dataset.append({
            'query': query,
            'reference': reference,
            'label': label
        })
    return dataset

def convert_dataset(examples, tokenizer, max_length, display_examples=False):
    """
    Convert train examples into BERT's input foramt.
    """
    features = []

    for ex_idx, example in enumerate(examples):
        sentence_a = tokenizer.tokenize(example['reference'])
        sentence_b = tokenizer.tokenize(example['query'])

        if len(sentence_a) + len(sentence_b) > max_length - 3:  # 3 for [CLS], 2x[SEP]
            logger.warning(
                "The length of the input is longer than max_length! "
                f"sentence_a: {sentence_a} / sentence_b: {sentence_b}"
            )
            # truncate sentence_b to fit in max_length
            diff = (len(sentence_a) + len(sentence_b)) - (max_length - 3)
            sentence_b = sentence_b[:-diff]
        
        tokens = ["[CLS]"] + sentence_a + ["[SEP]"] + sentence_b + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * (len(sentence_a) + 2) + [1] * (len(sentence_b) + 1)
        mask = [1] * len(input_ids)

        # Zero-padding
        padding = [0] * (max_length - len(input_ids))
        input_ids += padding
        segment_ids += padding
        mask += padding
        assert len(input_ids) == max_length
        assert len(segment_ids) == max_length
        assert len(mask) == max_length

        if ex_idx < 3 and display_examples:
            logger.info(f"========= Train Example {ex_idx+1} =========")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("mask: %s" % " ".join([str(x) for x in mask]))
            logger.info("label: %s" % example['label'])
            logger.info("")
        
        features.append({
            'input_ids': input_ids,
            'segment_ids': segment_ids,
            'input_mask': mask,
            'label': example['label']
        })
    return features

def build_rte_model(cache_dir, num_labels = 2):
    return BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', 
                                                          cache_dir=cache_dir,
                                                          num_labels=num_labels
                                                         )

def main():
    parser = argparse.ArgumentParser()

    # arguments
    parser.add_argument("--input_dir",
                        default="./data/",
                        type=str,
                        help="The input data dir."
    )
    parser.add_argument("--output_dir",
                        default="./rte/tmp/",
                        type=str,
                        help="The output dir where the model predictions will be stored."
    )
    parser.add_argument("--checkpoints_dir",
                        default="./rte/checkpoints/",
                        type=str,
                        help="Where checkpoints will be stored."
    )
    parser.add_argument("--cache_dir",
                        default="./data/models/",
                        type=str,
                        help="Where do you want to store the pre-trained models"
                        "downloaded from pytorch pretrained model."
    )
    parser.add_argument("--batchsize",
                        default=8,
                        type=int,
                        help="Batch size for training examples."
    )
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate."
    )
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--num_train_epochs",
                        default=10,
                        type=int,
                        help="Total number of training epochs."
    )
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="random seed."
    )
    parser.add_argument("--max_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after tokenized."
                        "If longer than this, it will be truncated, else will be padded."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.DEBUG)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    num_labels = 3  # [0 for SUPPORT, 1 for REFUTE, 2: NEI]
    criterion = CrossEntropyLoss()

    # Make sure to pass do_lower_case=False when use multilingual-cased model.
    # See https://github.com/google-research/bert/blob/master/multilingual.md
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',
                                              do_lower_case=False)
    model = build_rte_model(args.cache_dir, num_labels)

    # prepare the dataset
    train_dataset = get_dataset(args.input_dir, 'train')
    val_dataset = get_dataset(args.input_dir, 'test')

    # convert dataset into BERT's input formats
    train_features = convert_dataset(
        train_dataset, 
        tokenizer, 
        args.max_length,
        display_examples=False
    )
    val_features = convert_dataset(
        val_dataset,
        tokenizer,
        args.max_length,
        display_examples=False
    )

    # prepare optimizer
    num_train_optimization_steps = int(len(train_dataset) / args.batchsize) * args.num_train_epochs
    optimizer = BertAdam(model.parameters(), 
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    model.to(device)

    global_step = 0
    
    # prepare train dataset
    all_input_ids = torch.LongTensor([x['input_ids'] for x in train_features])
    all_segment_ids = torch.LongTensor([x['segment_ids'] for x in train_features])
    all_input_mask = torch.LongTensor([x['input_mask'] for x in train_features])
    all_label = torch.LongTensor([x['label'] for x in train_features])
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, 
        sampler=train_sampler, 
        batch_size=args.batchsize, 
        drop_last=True
    )

    # prepare validation dataset
    all_input_ids_eval = torch.LongTensor([x['input_ids'] for x in val_features])
    all_segment_ids_eval = torch.LongTensor([x['segment_ids'] for x in val_features])
    all_input_mask_eval = torch.LongTensor([x['input_mask'] for x in val_features])
    all_label_eval = torch.LongTensor([x['label'] for x in val_features])
    valid_data = TensorDataset(all_input_ids_eval, all_input_mask_eval, all_segment_ids_eval, all_label_eval)

    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(
        valid_data, 
        sampler=valid_sampler, 
        batch_size=args.batchsize, 
        drop_last=False
    )

    # TRAINING !!
    logger.info("=== Running training ===")
    logger.info("===== Num examples : %d", len(train_dataset))
    logger.info("===== Batch size : %d", args.batchsize)
    logger.info("===== Num steps : %d", num_train_optimization_steps)

    # training
    max_acc = 0
    for epoch in range(int(args.num_train_epochs)):
        model.train()
        tr_loss, num_tr_examples, num_tr_steps = 0, 0, 0
        temp_tr_loss, temp_num_tr_exs, temp_num_tr_steps = 0, 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, labels = batch

            model.zero_grad()
            # compute loss and backpropagate
            loss, logits = model(
                input_ids, 
                token_type_ids=segment_ids, 
                attention_mask=input_mask, 
                labels=labels
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            tr_loss += loss.item()
            num_tr_examples += input_ids.size(0)
            num_tr_steps += 1

            # logging every 0.05 epoch
            temp_tr_loss += loss.item()
            temp_num_tr_exs += input_ids.size(0)
            temp_num_tr_steps += 1
            if (step + 1) % (len(train_dataloader) // 1) == 0:
                logger.info("Epoch %d/%d - step %d/%d" % ((epoch+1), args.num_train_epochs, step, len(train_dataloader)))
                logger.info("# of examples %d" % temp_num_tr_exs)
                logger.info("temp loss %f" % (temp_tr_loss / temp_num_tr_steps))
                temp_tr_loss, temp_num_tr_exs, temp_num_tr_steps = 0, 0, 0
        
        # logging every 1 epoch
        print('===== Epoch %d done.' % (epoch+1))
        print('===== Average training loss', tr_loss / num_tr_steps)

        # validate every 1 epoch
        logger.info("=== Running validation ===")
        model.eval()
        eval_loss, eval_acc = 0, 0
        for batch in tqdm(valid_dataloader, desc="Iteration"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, labels = batch
            with torch.no_grad():
                loss, logits = model(
                    input_ids, 
                    token_type_ids=segment_ids, 
                    attention_mask=input_mask, 
                    labels=labels
                )
                
            eval_loss += loss.item()
            eval_acc += calculate_metric(
                logits.detach().cpu().numpy(),
                labels.to('cpu').numpy()    
            )
        eval_acc_ =  eval_acc / len(valid_dataloader)
        if max_acc < eval_acc_ :
            max_acc = eval_acc_
            torch.save({'epoch': epoch + 1,
                        'model_state': model.state_dict(),
                        'optimizer_state' : optimizer.state_dict()},
                        os.path.join(args.checkpoints_dir, 'best_ckpt.pth'))

        # logging validation results
        print('===== Validation loss', eval_loss / len(valid_dataloader))
        print('===== Validation accuracy', eval_acc / len(valid_dataloader))


if __name__ == "__main__":
    print(f"Job is running on {socket.gethostname()}")
    main()