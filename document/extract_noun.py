import json
import numpy as np
from tqdm import tqdm
from konlpy.tag import Kkma
kkma = Kkma()
from collections import defaultdict

with open('./data/claims_korquad.json', 'r') as f:
    wiki = json.load(f)
print(len(wiki))
# Remove duplicates
dup = {}
for doc in wiki:
    if doc['context'] not in dup:
        dup[doc['context']] = len(dup)
print(len(dup))
keys = sorted(list(dup.values()))
new_wiki = []
for key in tqdm(keys, ncols=80):
    new_wiki.append(wiki[key])
    new_wiki[-1]['kkma_nouns'] = kkma.nouns(new_wiki[-1]['context'])
    
with open('./data/docs_noun.json', 'w') as f:
    json.dump(new_wiki, f)