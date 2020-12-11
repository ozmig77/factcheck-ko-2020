import os
import json
import random
import argparse
import logging
import socket
import pandas as pd
import numpy as np
#import kss
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import (
    TensorDataset, 
    DataLoader, 
    RandomSampler,
)
from torch.nn.utils import clip_grad_norm_

from transformers import BertTokenizer, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam


logger = logging.getLogger(__name__)


def calculate_metric(logits, labels):
    softmax_logits = F.softmax(logits, 1)[:, 1]
    # accuracy
    acc = 1 if softmax_logits.argmax() == labels.argmax() else 0

    # r5
    temp = [(k, v) for k, v in zip(softmax_logits, labels)]
    temp.sort(key=lambda tup: tup[0], reverse=True)
    r5 = 1 if sum([x for _, x in temp[:5]]) == 1 else 0

    return acc, r5

def get_dataset(dir, split='train'):
    with open('data/{}_anno.json'.format(split), 'r') as f:
        jsondata = json.load(f)
    dataset = []
    for elem in jsondata:
        candidates = elem['context'].split('. ')
        labels = [0] * len(candidates)
        labels[candidates.index(elem['reference'])] = 1
        query = elem['paraphrased']
        dataset.append({
            'query': query,
            'candidates': candidates,
            'labels': labels
        })
    return dataset

def split_pos_neg_examples(dataset):
    examples_pos = []
    examples_neg = []
    for d in dataset:
        for label, c in zip(d['labels'], d['candidates']):
            example = {
                'query': d['query'],
                'candidate': c,
                'label': label
            }
            if label == 1:
                examples_pos.append(example)
            else:
                examples_neg.append(example)
    return examples_pos, examples_neg

def convert_train_dataset(examples, tokenizer, max_length, display_examples=False):
    """
    Convert train examples into BERT's input foramt.
    """
    features = []

    for ex_idx, example in enumerate(examples):
        sentence_a = tokenizer.tokenize(example['candidate'])
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
        
def convert_valid_dataset(examples, tokenizer, max_length):
    """
    Convert valid examples into BERT's input foramt.
    """
    features = []

    for ex_idx, example in enumerate(examples):
        input_ids_list = []
        segment_ids_list = []
        mask_list = []
        sentence_b = tokenizer.tokenize(example['query'])
        for c in example['candidates']:
            sentence_a = tokenizer.tokenize(c)

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

            input_ids_list.append(input_ids)
            segment_ids_list.append(segment_ids)
            mask_list.append(mask)

        features.append({
            'input_ids': input_ids_list,
            'segment_ids': segment_ids_list,
            'input_mask': mask_list,
            'label': example['labels']
        })

    return features


def build_ss_model(cache_dir, num_labels = 2):
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
                        default="./ss/tmp/",
                        type=str,
                        help="The output dir where the model predictions will be stored."
    )
    parser.add_argument("--checkpoints_dir",
                        default="./ss/checkpoints/",
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
                        default=4,
                        type=int,
                        help="Batch size for (positive) training examples."
    )
    parser.add_argument("--negative_batchsize",
                        default=4,
                        type=int,
                        help="Batch size for (negative) training examples."
    )
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate."
    )
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--num_train_epochs",
                        default=5,
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

    num_labels = 2
    criterion = CrossEntropyLoss()

    # Make sure to pass do_lower_case=False when use multilingual-cased model.
    # See https://github.com/google-research/bert/blob/master/multilingual.md
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',
                                              do_lower_case=False)
    model = build_ss_model(args.cache_dir, num_labels)

    # prepare the dataset
    train_dataset = get_dataset(args.input_dir, 'train')
    val_dataset = get_dataset(args.input_dir, 'test')
    
    # convert dataset into BERT's input formats
    train_examples_pos, train_examples_neg = split_pos_neg_examples(train_dataset)
    train_features = convert_train_dataset(
        train_examples_pos, 
        tokenizer, 
        args.max_length
    )
    train_features_neg = convert_train_dataset(
        train_examples_neg, 
        tokenizer, 
        args.max_length
    )
    val_features = convert_valid_dataset(
        val_dataset,
        tokenizer,
        args.max_length
    )
    
    # prepare optimizer
    num_train_optimization_steps = int(len(train_examples_pos) / args.batchsize) * args.num_train_epochs
    optimizer = BertAdam(model.parameters(), 
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    model.to(device)

    global_step = 0

    # TRAINING !!
    logger.info("=== Running training ===")
    logger.info("===== Num (pos) examples : %d", len(train_examples_pos))
    logger.info("===== Batch size : %d", args.batchsize)
    logger.info("===== Num steps : %d", num_train_optimization_steps)
    
    # prepare positive/negative train dataset
    all_input_ids = torch.LongTensor([x['input_ids'] for x in train_features])
    all_segment_ids = torch.LongTensor([x['segment_ids'] for x in train_features])
    all_input_mask = torch.LongTensor([x['input_mask'] for x in train_features])
    all_label = torch.LongTensor([x['label'] for x in train_features])
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)

    all_input_ids_neg = torch.LongTensor([x['input_ids'] for x in train_features_neg])
    all_segment_ids_neg = torch.LongTensor([x['segment_ids'] for x in train_features_neg])
    all_input_mask_neg = torch.LongTensor([x['input_mask'] for x in train_features_neg])
    all_label_neg = torch.LongTensor([x['label'] for x in train_features_neg])
    train_data_neg = TensorDataset(all_input_ids_neg, all_input_mask_neg, all_segment_ids_neg, all_label_neg)

    train_sampler = RandomSampler(train_data)
    train_sampler_neg = RandomSampler(train_data_neg)

    train_dataloader = DataLoader(
        train_data, 
        sampler=train_sampler, 
        batch_size=args.batchsize, 
        drop_last=True
    )
    negative_dataloader = DataLoader(
        train_data_neg, 
        sampler=train_sampler_neg, 
        batch_size=args.negative_batchsize, 
        drop_last=True
    )
    # training
    max_acc = 0
    for epoch in range(int(args.num_train_epochs)):
        model.train()
        tr_loss, num_tr_examples, num_tr_steps = 0, 0, 0
        temp_tr_loss, temp_num_tr_exs, temp_num_tr_steps = 0, 0, 0
        it = iter(negative_dataloader)
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            batch_neg = tuple(t.to(device) for t in next(it))

            input_ids, input_mask, segment_ids, labels = batch
            input_ids_neg, input_mask_neg, segment_ids_neg, labels_neg = batch_neg

            # batchify
            input_ids_cat=torch.cat([input_ids, input_ids_neg],dim=0)
            segment_ids_cat=torch.cat([segment_ids, segment_ids_neg],dim=0)
            input_mask_cat=torch.cat([input_mask,input_mask_neg],dim=0)
            label_ids_cat=torch.cat([labels.view(-1), labels_neg.view(-1)], dim = 0)

            model.zero_grad()
            # compute loss and backpropagate
            loss, logits = model(
                input_ids_cat, 
                token_type_ids=segment_ids_cat, 
                attention_mask=input_mask_cat, 
                labels=label_ids_cat
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
            if (step + 1) % (len(train_dataloader) // 20) == 0:
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
        eval_loss, eval_acc, eval_r5 = 0, 0, 0
        for example in tqdm(val_features, desc="Iteration"):
            input_ids = torch.LongTensor(example['input_ids']).to(device)
            segment_ids = torch.LongTensor(example['segment_ids']).to(device)
            input_mask = torch.LongTensor(example['input_mask']).to(device)
            label = torch.LongTensor(example['label']).to(device)

            with torch.no_grad():
                loss, logits = model(
                    input_ids, 
                    token_type_ids=segment_ids, 
                    attention_mask=input_mask, 
                    labels=label
                )
                
            eval_loss += loss.item()
            temp_acc, temp_r5 = calculate_metric(logits, label)
            eval_acc += temp_acc
            eval_r5 += temp_r5
        eval_acc_ =  eval_acc / len(val_features)
        if max_acc < eval_acc_ :
            max_acc = eval_acc_
            torch.save({'epoch': epoch + 1,
                        'model_state': model.state_dict(),
                        'optimizer_state' : optimizer.state_dict()},
                        os.path.join(args.checkpoints_dir, 'best_ckpt.pth'))

        # logging validation results
        print('===== Validation loss', eval_loss / len(val_features))
        print('===== Validation accuracy', eval_acc / len(val_features))
        print('===== Validation R@5', eval_r5 / len(val_features))


if __name__ == "__main__":
    print(f"Job is running on {socket.gethostname()}")
    main()