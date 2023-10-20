import evaluate
import nltk
import numpy as np
import pickle
import datasets
import numpy as np
import re

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

ds_dir = '' # dataset directory
model_dir = '' # output from the model
colname = '' # input column (knowledge source / factual corpus) name 

ds = datasets.load_dataset(ds_dir)
rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')

for res_name, ds, colname in [(model_dir, ds, colname)]:

    ds = ds['test']

    print("\n=================\nDATASET:", model_dir, '\n')

    with open(model_dir, 'rb') as handle:
        model = pickle.load(handle)

    source = ds['template']

    def calc_total_hallucinations(pred, source):

        pred_tok, source_tok = [nltk.word_tokenize(p) for p in pred], [[nltk.word_tokenize(p_) for p_ in p] for p in source]
        source_tok = [[item for sublist in p for item in sublist] for p in source_tok]

        source_tok = [[re.sub(r'\W+', '', w).lower() for w in tok] for tok in source_tok]
        pred_tok = [[re.sub(r'\W+', '', w).lower() for w in tok] for tok in pred_tok]
        
        hall = []
        for i in range(len(pred_tok)):

            source_set = set(source_tok[i])
            
            num_halluc = 0
            total_tok = 0

            for word in pred_tok[i]:
                num_halluc += int(word not in source_set)
                total_tok += 1

            if len(pred_tok[i]) == 0:
                hall.append(1)
                continue

            if total_tok == 0:
                hall.append(0)
            else:
                hall.append((1.0 * num_halluc) / (total_tok))

        return np.mean(np.array(hall)) * 100

    def evaluate_metrics(pred, true, source):
        true = [t.lower() for t in true]
        pred = [p.lower() for p in pred]

        halluc = calc_total_hallucinations(pred, source)

        rouge_vals = rouge.compute(predictions=pred, references=true)
        bleu_vals = bleu.compute(predictions=pred, references=true)
        meteor_vals = meteor.compute(predictions=pred, references=true)
        
        avg_length = np.mean(np.array([len(nltk.word_tokenize(doc)) for doc in pred]))
        
        print(f"Rouge:\n{rouge_vals}\n\nBleu:\n{bleu_vals}\n\nMeteor:{meteor_vals}\n\nLength: {avg_length}\n\nHallucinations: {halluc}")

    evaluate_metrics(model, ds['output'], sources)
