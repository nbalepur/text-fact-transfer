device = 'cuda:0' # which cuda device to use
ds_dir = '' # dataset directory
out_dir = '' # output directory
q_gen_model_name = '' # name of QG model
qa_model_name = '' # name of QA model

import datasets
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForQuestionAnswering
q_gen_model = AutoModelForSeq2SeqLM.from_pretrained(q_gen_model_name).to(device)
q_gen_tokenizer = AutoTokenizer.from_pretrained(q_gen_model_name)

answer_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name).to(device)
answer_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

import torch
import time
from transformers import AutoTokenizer, AutoModel
retriever_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
retriever_model = AutoModel.from_pretrained('facebook/contriever-msmarco').to(device)

import nltk
import numpy as np

import torch
import gc

def generate_questions(title, context):

    data = []

    for sent in nltk.sent_tokenize(context):

        input_text = f"<|title|> {title} <|context|> {sent}"
        in_data = q_gen_tokenizer(input_text, max_length=1024, truncation=True, padding='max_length', return_tensors='pt')
        input_ids, attn_mask = in_data.input_ids.to(device), in_data.attention_mask.to(device)
        res = q_gen_model.generate(input_ids, attention_mask = attn_mask, max_length=128, do_sample=True, top_p=0.75, num_return_sequences=10)

        out = q_gen_tokenizer.batch_decode(res)

        for out_ in out:
            curr_data = []
            curr_out = out_.replace('</s>', '').replace('<s>', '').replace('<pad>', '')
            curr_tag = "<|answer|>"
            while curr_tag in curr_out:
                new_idx = curr_out.index(curr_tag) + len(curr_tag)
                curr_str = curr_out[:new_idx].replace(curr_tag, "").strip().replace(".", "")
                curr_out = curr_out[new_idx:]
                curr_tag = "<|answer|>" if curr_tag != "<|answer|>" else "<|question|>"
                if len(curr_data) == 0 and curr_tag == "<|question|>":
                    continue
                if len(curr_data) != 0 and len(curr_data[-1]) == 1:
                    curr_data[-1].append(curr_str)
                else:
                    curr_data.append([curr_str])

            if len(curr_data[-1]) == 1:
                curr_data[-1].append(curr_out.strip())

            if sum([len(x) != 2 for x in curr_data]) != 0:
                print(curr_data)
                print('\n\n')
                print(curr_out)
                print('\n\n')
                print(out_)
                exit(1)

            #print(curr_data)

            data.extend(curr_data)

    qa_map = dict()
    for a, q in data:
        if (a not in context and a.replace(".", "") not in context.replace(".", "")) or a.lower() == title.lower():
            continue
        curr_set = qa_map.get(a, set())
        curr_set.add(q)
        qa_map[a] = curr_set

    keys = list(qa_map.keys())
    keys_to_pop = set()
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i == j:
                continue
            if keys[i].lower() in keys[j].lower():
                keys_to_pop.add(keys[i])


    for key in keys_to_pop:
        qa_map.pop(key)

        
    return qa_map

def get_old_answers(questions, contexts):

    output = []
    for i in range(len(questions)):
        QA_input = {
            'question': questions[i],
            'context': contexts[i]
        }
        res = nlp(QA_input)
        output.append(res['answer'])
    
    return output

import itertools
def get_new_answers_ext(questions_, corpus, corpus_embed, ref_answer, k, title, should_emb):

    combo = list(itertools.product(questions_, repeat=2))
    questions = [c[0] for c in combo]
    retr_questions = [c[1] for c in combo]
    
    #print(retr_questions)
    
    if should_emb:
        contexts = []
        for query in retr_questions:
            query_embed = mean_pooling(query)
            context = [corpus[idx] for idx in torch.argsort(-1 * (corpus_embed @ query_embed.T).squeeze(1))[:k]]
            contexts.append('\n'.join(context))
    else:
        contexts = [corpus for _ in retr_questions]
    
    #print(contexts[0], '\n\n')
    examples = {'question': questions, 'ref_answer': [ref_answer.lower() for _ in questions], 'context': contexts}

    inputs_ = ["<|question|> " + examples['question'][i] + " <|answer|> " + examples['ref_answer'][i] + " <|context|> " + examples['context'][i] for i in range(len(examples["context"]))]
    
    start_logits, end_logits = [], []
    input_ids2 = []
    for inputs in inputs_:
        inputs = answer_tokenizer(inputs, max_length=256, truncation=True, padding='max_length', return_tensors='pt').to(device)
        input_ids2.append(inputs.input_ids.to('cpu').detach())
        res = answer_model(inputs.input_ids, attention_mask = inputs.attention_mask)
        start_logits_, end_logits_ = res.start_logits.to('cpu').detach(), res.end_logits.to('cpu').detach()
        start_logits.append(start_logits_)
        end_logits.append(end_logits_)
        
    input_ids2 = torch.cat(input_ids2, axis = 0)
    start_logits1, end_logits1 = torch.cat(start_logits, axis = 0), torch.cat(end_logits, axis = 0)
        
    max_len = max(15, 2 * len(answer_tokenizer(ref_answer).input_ids))
    
    all_answers = []
    for i in range(len(contexts)):
        start_logits, end_logits = start_logits1[i], end_logits1[i]
        start_indexes = np.argsort(start_logits).tolist()
        end_indexes = np.argsort(end_logits).tolist()
        
        valid_answers = []
        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index <= end_index and end_index - start_index <= max_len: # We need to refine that test to check the answer is inside the context
                    answer = answer_tokenizer.decode(input_ids2[i][start_index:end_index])
                    if answer.lower() != title.lower() and len(answer) > 2 and '<|question|>' not in answer and '<|answer|>' not in answer and '<|context|>' not in answer:
                        valid_answers.append((
                            start_logits[start_index] + end_logits[end_index].item(),
                            answer
                            ))
        best_val = max(valid_answers, key = lambda item: item[0])
        all_answers.append((best_val[1], best_val[0]))
    
    return all_answers

def cos_sim(w1, w2):
    emb1, emb2 = mean_pooling(w1), mean_pooling(w2)
    return (emb1 @ emb2.T) / (torch.sum(emb1 * emb1) * torch.sum(emb2 * emb2))**0.5

def mean_pooling(sentence):
    inputs = retriever_tokenizer([sentence], padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = retriever_model(**inputs)[0]
    token_embeddings, mask = outputs, inputs['attention_mask']
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = (token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]).to('cpu').detach()
    return sentence_embeddings

def best_answer(cand_answers):
    all_answers = {}
    for cand, s in cand_answers:
        score = all_answers.get(cand.lower(), 0.0)
        all_answers[cand.lower()] = (score + s)
    return max(all_answers, key = lambda item: all_answers[item])

def query_expansion(query, old_topic, new_topic):
    query = query.lower()
    spec_q = query.replace(old_topic.lower(), new_topic.lower())
    gen_q = ' '.join([w for w in nltk.word_tokenize(query) if w in nltk.word_tokenize(spec_q)])
    
    return [spec_q, gen_q]

import requests
import unidecode
from bs4 import BeautifulSoup
def web_search(query, num_results, banned_word):

    search = query.replace(' ', '+')
    url = f"https://www.google.com/search?q={search}&num={num_results}"

    requests_results = requests.get(url)
    if requests_results.status_code == 429:
        print(429)
    time.sleep(15 + np.random.uniform())
    
    soup = BeautifulSoup(requests_results.content, "html.parser")
    #print(soup)
    ret = soup.find_all('div', attrs={'class': 'BNeawe s3v9rd AP7Wnd'})
    
    texts = []
    for r in ret:
        try:
            texts.append(r.find_all('div')[0].find_all('div')[0].find_all('div')[0].text)
        except:
            continue
    #print('\n\n'.join(texts))
    
    links = soup.find_all("a")

    urls = []
    for link in links:
        link_href = link.get('href')
        if "url?q=" in link_href and not "webcache" in link_href:
            title = link.find_all('h3')
            if len(title) > 0:
                urls.append(link.get('href').split("?q=")[1].split("&sa=U")[0])
                
    texts = [unidecode.unidecode(texts[i]) for i in range(min(len(texts), len(urls))) if banned_word not in urls[i]]
    return '\n'.join(texts)

def merge_template(template, template_title, title, entities, answers): # merge new info into template
    template = template.replace(", #", ",#").replace(" - ", "-").replace(" .", ".").replace("( ", "(").replace(" )", ")").replace(" '", "'").replace("$ ", "$").replace("# ", "#").strip().lower()
    template = template.replace(template_title.lower(), title.lower())
    for i, a in enumerate(answers):
        template = template.replace(entities[i].lower(), a)
    return template

import tqdm

should_clean = True
should_emb = True
num_corpus_retr = 5

# --------------------------------- model inference ---------------------------------

# initialize parameters for run
ds_ = datasets.load_dataset(ds_dir)
corpus_name = 'corpus'
res_name = 'google'

no_entities_ids = []

aug_data = dict()

for x in ds_.keys():

    ds = ds_[x]
    out1 = []

    for i in tqdm.tqdm(range(len(ds['template']))):
        # load data
        template, template_title = ds['template'][i], ds['template_title'][i].title()
        corpus = ds[corpus_name][i]

        #print(template, template_title)

        #print('retrieval embeddings')
        title = ds['title'][i]
        if should_emb:
            corpus_embed = torch.cat([mean_pooling(sent) for sent in corpus], axis = 0)
        else:
            corpus_embed = None

        # generate questions
        #print('generate questions')
        qa_map = generate_questions(template_title, template) # get questions

        #print(qa_map)

        #print('answer questions')
        answers_with_desc = []
        entities = []
        for ans, questions in qa_map.items():
            questions_ = [query_expansion(q, template_title, title) for q in questions]
            questions_ = [item for sublist in questions_ for item in sublist]
            entities.append(ans)
            cand_answers = get_new_answers_ext(list(questions_), corpus, corpus_embed, ans, num_corpus_retr, title, should_emb)
            answers_with_desc.append(best_answer(cand_answers).replace(", ", ",").replace(" - ", "-").replace(" .", ".").replace("( ", "(").replace(" )", ")").replace(" '", "'").replace("$ ", "$").replace("# ", "#").strip())

        if len(entities) == 0:
            print(i)
            no_entities_ids.append(i)

        out1.append((entities, qa_map, questions_, answers_with_desc))

    aug_data[x] = out1

import pickle
with open(out_dir, 'wb') as handle:
    pickle.dump(aug_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
