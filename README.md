# Factcheck-ko-2020

### Install
`git clone https://github.com/ozmig77/factcheck-ko-2020`
`pip install -r requirements.txt`
pretrained checkpoints link is given [here](https://drive.google.com/file/d/1JcMYXS0wSxAE3Rm-TMulu6a4nrhllSvT/view?usp=sharing).
save it in `ss/checkpoints` and `rte/checkpoints`.

### Datas
- `data/claims_korquad.json`: Supporting document for the Factcheck (Based on ko wiki) 
- `data/{train, val, test}.json`: Annotated dataset for the Factcheck

## Training

### Document Retrieval
For DR, run `python document/extract_noun.py`.
this will create `data/docs_noun.json`.

### Sentence Selection
For training SS, run `python ss/train_ss.py`.
this will create `ss/checkpoints/best_ckpt.pth`.

### Recognizing Textual Entailment
For training RTE, run `python rte/train_rte.py`.
this will create `rte/checkpoints/best_ckpt.pth`.

## Evaluation
For evaluation, run `python eval.py --k 5`.
It will take about 100 minute to process 4459 test claims, you can reduce time by setting k as 1.
Ours result R@5 94.93%, R@1 95.38% on SS, 64.21% on RTE.

Check `demo.ipynb` for demo.