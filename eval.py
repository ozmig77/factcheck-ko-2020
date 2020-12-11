import json
import pickle
import os
import warnings
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
# For extract noun
from konlpy.tag import Kkma
kkma = Kkma() 
import pandas as pd
import nltk
#import kss 

from ss.train_ss import build_ss_model
from rte.train_rte import build_rte_model
from transformers import BertTokenizer

      
class SS_Dataset(torch.utils.data.Dataset):
    def __init__(self, anno_json, docs, tokenizer, max_length, k):
        """
        Convert valid examples into BERT's input foramt.
        """
        self.max_length = max_length
        self.tokenizer = tokenizer
        # Sentence tokenization for documents
        #for doc in docs:
        #    sentences = kss.split_sentences(doc['context'])  # Korean Sentence Splitter(kss)
        #    doc['sentences'] = sentences

        print('Process SS dataloader')
        self.data = []
        for vidx, d in enumerate(anno_json):
            candidates = []
            for didx in d['dr_result'][:k]:
                candidates += docs[didx]['context'].split('. ')
            for c in candidates:
                self.data.append((vidx, d['paraphrased'], c))
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vidx, query, cand = self.data[idx]
        meta = {'vidx': vidx, 'cand': cand}
        
        sentence_b = self.tokenizer.tokenize(query)
        sentence_a = self.tokenizer.tokenize(cand)
        if len(sentence_a) + len(sentence_b) > self.max_length - 3:  # 3 for [CLS], 2x[SEP]
            #print(
            #    "The length of the input is longer than max_length! "
            #    f"sentence_a: {sentence_a} / sentence_b: {sentence_b}"
            #)
            # truncate sentence_b to fit in max_length
            diff = (len(sentence_a) + len(sentence_b)) - (self.max_length - 3)
            sentence_a = sentence_a[:-diff]
        
        tokens = ["[CLS]"] + sentence_a + ["[SEP]"] + sentence_b + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * (len(sentence_a) + 2) + [1] * (len(sentence_b) + 1)
        input_mask = [1] * len(input_ids)

        # Zero-padding
        padding = [0] * (self.max_length - len(input_ids))
        input_ids += padding
        segment_ids += padding
        input_mask += padding
        assert len(input_ids) == self.max_length
        assert len(segment_ids) == self.max_length
        assert len(input_mask) == self.max_length

        return input_ids, segment_ids, input_mask, meta

def collate_fn(batch):
    collections = list(zip(*batch))
    for i in range(3):
        collections[i] = torch.LongTensor(collections[i])
    return collections

class RTE_Dataset(torch.utils.data.Dataset):
    def __init__(self, anno_json, tokenizer, max_length, k):
        """
        Convert valid examples into BERT's input foramt.
        """
        self.max_length = max_length
        self.tokenizer = tokenizer

        print('Process RTE dataloader')
        self.data = []
        for vidx, d in enumerate(anno_json):
            for sidx, (_, sent) in enumerate(d['ss_result'][:k]):
                self.data.append((vidx, d['paraphrased'], sent, sidx))
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vidx, query, cand, sidx = self.data[idx]
        meta = {'vidx': vidx, 'sidx': sidx}
        
        sentence_b = self.tokenizer.tokenize(query)
        sentence_a = self.tokenizer.tokenize(cand)
        if len(sentence_a) + len(sentence_b) + 3 > self.max_length:  # 3 for [CLS], 2x[SEP]
            #print(
            #    "The length of the input is longer than max_length! "
            #    f"sentence_a: {sentence_a} / sentence_b: {sentence_b}"
            #)
            # truncate sentence_b to fit in max_length
            diff = (len(sentence_a) + len(sentence_b) + 3) - self.max_length
            sentence_a = sentence_a[:-diff]
        
        tokens = ["[CLS]"] + sentence_a + ["[SEP]"] + sentence_b + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * (len(sentence_a) + 2) + [1] * (len(sentence_b) + 1)
        input_mask = [1] * len(input_ids)

        # Zero-padding
        padding = [0] * (self.max_length - len(input_ids))
        input_ids += padding
        segment_ids += padding
        input_mask += padding
        assert len(input_ids) == self.max_length
        assert len(segment_ids) == self.max_length
        assert len(input_mask) == self.max_length

        return input_ids, segment_ids, input_mask, meta

# --- Document retrieval ---------------------
def document_retrieval(args, docs, anno_json):
    result = []
    for idx in tqdm(range(len(anno_json))):
        paraphrased = anno_json[idx]['paraphrased']
        NNs = set(kkma.nouns(paraphrased))
        count = defaultdict(int)
        for didx, doc in enumerate(docs):
            ctx = doc['context']
            ctx_set = set(doc['kkma_nouns'])
            for nn in NNs:
                if nn in ctx_set:
                    count[didx] += 1
        count_list = list(count.items())
        count_list.sort(key=lambda x: -1*x[1])
        anno_json[idx]['dr_result'] = [didx for didx, _ in count_list[:5]]

def sentence_selection(args, docs, anno_json, tokenizer):
    dataset = SS_Dataset(
        anno_json, 
        docs,
        tokenizer,
        args.max_length,
        args.k
    )
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn= collate_fn
        )
    
    print('Load checkpoint for SS')
    ss_model = build_ss_model(args.cache_dir, num_labels=2)
    ckpt = torch.load(os.path.join(args.ss_dir, 'best_ckpt.pth'))
    ss_model.load_state_dict(ckpt['model_state'])
    ss_model.cuda()
    ss_model.eval()
    
    results = defaultdict(list)
    for input_ids, segment_ids, input_mask, metas in tqdm(dataloader, ncols=80):
        input_ids = input_ids.cuda()
        segment_ids = segment_ids.cuda()
        input_mask = input_mask.cuda()

        with torch.no_grad():
            logits, = ss_model(
                input_ids, 
                token_type_ids=segment_ids, 
                attention_mask=input_mask, 
            )
            logits = F.softmax(logits, -1)
        logit_cpu = logits.cpu()
        for bidx, meta in enumerate(metas):
            results[meta['vidx']].append((logit_cpu[bidx, 1].item(), meta['cand']))

    # Save result
    for vidx, d in enumerate(anno_json):
        results[vidx].sort(key= lambda x: -1*x[0])
        d['ss_result'] = results[vidx][:5]
        
            
def rte(args, anno_json, tokenizer):
    dataset = RTE_Dataset(
        anno_json, 
        tokenizer,
        args.max_length,
        args.k
    )
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=2,
            drop_last=False,
            collate_fn= collate_fn
        )
    
    print('Load checkpoint for RTE')
    rte_model = build_rte_model(args.cache_dir, num_labels=3)
    ckpt = torch.load(os.path.join(args.rte_dir, 'best_ckpt.pth'))
    rte_model.load_state_dict(ckpt['model_state'])
    rte_model.cuda()
    rte_model.eval()
    
    results = defaultdict(list)
    for input_ids, segment_ids, input_mask, metas in tqdm(dataloader, ncols=80):
        input_ids = input_ids.cuda()
        segment_ids = segment_ids.cuda()
        input_mask = input_mask.cuda()
        with torch.no_grad():
            logits, = rte_model(
                input_ids, 
                token_type_ids=segment_ids, 
                attention_mask=input_mask, 
            )
            logits = F.softmax(logits, -1)
        logit_cpu = logits.cpu()
        for bidx, meta in enumerate(metas):
            results[meta['vidx']].append((logit_cpu[bidx], meta['sidx']))
    # Save results
    for vidx, d in enumerate(anno_json):
        d['rte_result'] = results[vidx]
        
def main(args):
    warnings.filterwarnings("ignore")
    # Load documents
    with open('./data/docs_noun.json', 'r') as f:
        json_docs = json.load(f)
    # prepare the dataset
    with open('data/test_anno.json', 'r') as f:
        val_json = json.load(f)
    tmpdir = 'tmp'
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    
    # ------ Retrieve documents ------------------
    if os.path.exists(tmpdir+'/eval_dr.json'):
        with open(tmpdir+'/eval_dr.json', 'r') as f:
            val_json = json.load(f)
    else:
        document_retrieval(args, json_docs, val_json)
        with open(tmpdir+'/eval_dr.json', 'w') as f:
            json.dump(val_json, f)
    # Calculate recall
    rank = []
    for vidx, d in enumerate(val_json):
        reference = d['context'].split(' ')
        rank.append(99999)
        for i, didx in enumerate(d['dr_result']):
            hypothesis = json_docs[didx]['context'].split(' ')
            #if json_docs[didx]['context'] == d['context']:
            BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.5, 0.5))
            if BLEUscore > 0.9:
                rank[-1] = i+1
                break
    recall5 = sum([1 for x in rank if x <= 5])
    recall1 = sum([1 for x in rank if x <= 1])
    print('DR R@1', recall1 / len(val_json), 'R@5', recall5/len(val_json)) 
    
    # ------ SS ---------------------------------
    # Make sure to pass do_lower_case=False when use multilingual-cased model.
    # See https://github.com/google-research/bert/blob/master/multilingual.md
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',
                                              do_lower_case=False)
    if os.path.exists(tmpdir+'/eval_ss.json'):
        with open(tmpdir+'/eval_ss.json', 'r') as f:
            val_json = json.load(f)
    else:
        sentence_selection(args, json_docs, val_json, tokenizer)
        with open(tmpdir+'/eval_ss.json', 'w') as f:
            json.dump(val_json, f)
    # Calculate recall
    rank = []
    for vidx, d in enumerate(val_json):
        reference = d['reference'].split(' ')
        for i, (_, sent) in enumerate(d['ss_result']):
            hypothesis = sent.split(' ')
            #if sent == d['reference']:
            BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.5, 0.5))
            if BLEUscore > 0.8:
                rank.append(i+1)
                break
    recall5 = sum([1 for x in rank if x <= 5])
    recall1 = sum([1 for x in rank if x <= 1])
    print('SS R@1', recall1 / len(val_json), 'R@5', recall5/len(val_json))    
    
    # ------ RTE ------------------------------------ 
    if os.path.exists(tmpdir+'/eval_rte.pkl'):
        with open(tmpdir+'/eval_rte.pkl', 'rb') as f:
            val_json = pickle.load(f)
    else:
        rte(args, val_json, tokenizer)
        with open(tmpdir+'/eval_rte.pkl', 'wb') as f:
            pickle.dump(val_json, f)
    # Calculate accuracy
    name2label = {'TRUE':0, 'FALSE':1, 'NEI':2}
    acc = []
    for vidx, d in enumerate(val_json):
        gt = name2label[d['True_False']]
        pred, norm = 0, 0
        if len(d['rte_result']) == 0:
            # No retrieved document in document retrieval
            acc.append(0)
            continue
        for rte_logit, sidx in d['rte_result']:
            pred += d['ss_result'][sidx][0] * rte_logit
            norm += d['ss_result'][sidx][0]
        pred = (pred / norm).argmax(0)
        acc.append(float(pred == gt))
    print('RTE Acc', sum(acc) / len(acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # arguments
    parser.add_argument("--input_dir",
                        default="./data/",
                        type=str,
                        help="The input data dir."
    )
    parser.add_argument("--ss_dir",
                        default="./ss/checkpoints/",
                        type=str,
                        help="The checkpoint dir for ss"
    )
    parser.add_argument("--rte_dir",
                        default="./rte/checkpoints/",
                        type=str,
                        help="The checkpoint dir for rte"
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
                        help="Batch size for (positive) training examples."
    )
    parser.add_argument("--k",
                        default=5,
                        type=int,
                        help="Size of retrieval"
    )
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="random seed."
    )
    parser.add_argument("--max_length",
                        default=510,
                        type=int,
                        help="The maximum total input sequence length after tokenized."
                        "If longer than this, it will be truncated, else will be padded."
    )
    args = parser.parse_args()
    
    main(args)
