import json
import pickle
import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
# For extract noun
from konlpy.tag import Kkma
import pandas as pd
import nltk
#import kss 

from ss.train_ss import build_ss_model
from rte.train_rte import build_rte_model
from transformers import BertTokenizer


class SS_Dataset(torch.utils.data.Dataset):
    def __init__(self, query, docs, tokenizer, max_length=512):
        """
        Convert valid examples into BERT's input foramt.
        """
        self.max_length = max_length
        self.tokenizer = tokenizer

        print('Process SS dataloader')
        self.data = []
        candidates = []
        for doc in docs:
            candidates += doc.split('. ')
        for c in candidates:
            self.data.append((query, c))
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, cand = self.data[idx]
        meta = {'cand': cand}        
        
        sentence_b = self.tokenizer.tokenize(query)
        sentence_a = self.tokenizer.tokenize(cand)
        if len(sentence_a) + len(sentence_b) > self.max_length - 3:  # 3 for [CLS], 2x[SEP]
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
    
class Factcheck:
    def __init__(self):
        with open('./data/docs_noun.json', 'r') as f:
            json_docs = json.load(f)
        self.docs = json_docs
        self.kkma = Kkma() 
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',
                                                       do_lower_case=False)
        self.max_length = 512
        self.cache_dir = './data/models/'
        
        print('Load checkpoint for SS')
        ss_model = build_ss_model(self.cache_dir, num_labels=2)
        ckpt = torch.load('./ss/checkpoints/best_ckpt.pth')
        ss_model.load_state_dict(ckpt['model_state'])
        ss_model.cuda()
        ss_model.eval()
        self.ss_model = ss_model
        
        print('Load checkpoint for RTE')
        rte_model = build_rte_model(self.cache_dir, num_labels=3)
        ckpt = torch.load('./rte/checkpoints/best_ckpt.pth')
        rte_model.load_state_dict(ckpt['model_state'])
        rte_model.cuda()
        rte_model.eval()
        self.rte_model = rte_model
        
    def document_retrieval(self, sent):
        '''
        params
          sent: string, sentence to factcheck
        output
          docs: list of string, evidence document
        '''
        NNs = set(self.kkma.nouns(sent))
        count = defaultdict(int)
        for didx, doc in enumerate(self.docs):
            ctx = doc['context']
            ctx_set = set(doc['kkma_nouns'])
            for nn in NNs:
                if nn in ctx_set:
                    count[didx] += 1
        count_list = list(count.items())
        count_list.sort(key=lambda x: -1*x[1])
        dids = [didx for didx, _ in count_list[:1]] # Select top1
        return dids
    
    def select_sentence(self, query, docs):
        print("Processing evidence retrieval")
        # Build data
        dataset = SS_Dataset(
            query, 
            docs,
            self.tokenizer,
            self.max_length
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            drop_last=False,
            collate_fn= collate_fn
        )
        
        results = []
        for input_ids, segment_ids, input_mask, metas in dataloader:
            input_ids = input_ids.cuda()
            segment_ids = segment_ids.cuda()
            input_mask = input_mask.cuda()

            with torch.no_grad():
                logits, = self.ss_model(
                    input_ids, 
                    token_type_ids=segment_ids, 
                    attention_mask=input_mask, 
                )
                logits = F.softmax(logits, -1)
            logit_cpu = logits.cpu()
            for bidx, meta in enumerate(metas):
                results.append((logit_cpu[bidx, 1].item(), meta['cand']))
        results.sort(key= lambda x: -1*x[0])
        return results[:2] # [(float, string)]
    
    def rte(self, query, evidence):
        print("Recognizing entailment")
        sentence_b = self.tokenizer.tokenize(query)
        sentence_a = self.tokenizer.tokenize(evidence)
        if len(sentence_a) + len(sentence_b) + 3 > self.max_length:  # 3 for [CLS], 2x[SEP]
            # truncate sentence_b to fit in max_length
            diff = (len(sentence_a) + len(sentence_b) + 3) - self.max_length
            sentence_a = sentence_a[:-diff]
        tokens = ["[CLS]"] + sentence_a + ["[SEP]"] + sentence_b + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * (len(sentence_a) + 2) + [1] * (len(sentence_b) + 1)
        input_mask = [1] * len(input_ids)
        
        input_ids = torch.LongTensor(input_ids).cuda().unsqueeze(0)
        segment_ids = torch.LongTensor(segment_ids).cuda().unsqueeze(0)
        input_mask = torch.LongTensor(input_mask).cuda().unsqueeze(0)
        with torch.no_grad():
            logits, = self.rte_model(
                input_ids, 
                token_type_ids=segment_ids, 
                attention_mask=input_mask, 
            )
            logits = F.softmax(logits, -1)
        logit_cpu = logits.cpu()
        return logit_cpu
    
    def check(self, query):
        dids = self.document_retrieval(query)
        docs = [self.docs[did]['context'] for did in dids]
        sents = self.select_sentence(query, docs)
        pred_logit = self.rte(query, sents[0][1])
        return self.docs[dids[0]], sents, pred_logit
    
if __name__ == '__main__':
    factcheck = Factcheck()
    docs, sents, pred_logit = factcheck.check("류현진은 야구선수이다")
    label2name = ['TRUE', 'FALSE', 'NEI']
    pred_class = pred_logit[0].argmax().item()
    print(docs)
    print("Retrived Evidence (Top 2)")
    for _, sent in sents:
        print(sent)
    print("Prediction: ", label2name[pred_class], "logit:, ", pred_logit[0][pred_class].item())