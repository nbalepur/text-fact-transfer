dataset_dir = ''

import datasets
ds = datasets.load_from_disk(dataset_dir)

import requests
from bs4 import BeautifulSoup
import re
import unidecode
import urllib.request as urllib2
import nltk
import time


def get_urls(query, num_results):

    search = query.replace(' ', '+')
    url = f"https://www.google.com/search?q={search}&num={2 * num_results}"
    time.sleep(10)
    requests_results = requests.get(url)

    soup_link = BeautifulSoup(requests_results.content, "html.parser")

    if 'captcha' in str(soup_link).lower():
        print('caught')
        print(soup_link)

    links = soup_link.find_all("a")

    ret = []
    
    for link in links:
        link_href = link.get('href')
        if "url?q=" in link_href and not "webcache" in link_href:
            title = link.find_all('h3')
            if len(title) > 0:
                ret.append(link.get('href').split("?q=")[1].split("&sa=U")[0])
    
    ret = ret[:num_results]
    return ret

def get_sentences_web(num_results, ret):

    sentences = []
    num_seen = 0
    for url in ret:
        
        if num_seen == num_results:
            break
        
        try:
            
            time.sleep(10)

            hdr = {'User-Agent': 'Mozilla/5.0'}
            req = urllib2.Request(url, headers = hdr)
            page = urllib2.urlopen(req, timeout = 3)

            if 'pdf' in page.headers['Content-Type'].lower():
                continue

            soup = BeautifulSoup(page, "html.parser", from_encoding="iso-8859-1")

            paragraphs_web = soup.findAll("p")
            paragraphs_web = [p.text for p in paragraphs_web]

            if len(paragraphs_web) > 200 or len(paragraphs_web) == 0:
                continue

            old_sentence_length = len(sentences)
            for p in paragraphs_web:
                curr_sentences = nltk.sent_tokenize(p)
                for sentence in curr_sentences:
                    if len(sentence) < 30 or "©" in sentence or "license" in sentence.lower() or "cookies" in sentence.lower() or "http" in sentence.lower():
                        continue
                    sentences.append(sentence) 
            if len(sentences) != old_sentence_length:
                num_seen += 1
        except Exception as e:
            _ = 1
            
    return sentences

def clean_text(text):
    
    if "displaystyle" in text:
        return ""
    
    text = re.sub(r'[^A-Za-z0-9.,?!:;\'\- ]+', ' ', text)
    text = text.replace('displaystyle', ' ').replace('\n', '')
    return re.sub(' +', ' ', text)


c = {}
import tqdm
import pickle
import random

for split in ds.keys():
    
    titles = ds[split]['title']
    out = []
    for title in tqdm.tqdm(titles):
        QUERY = f'{title} wikipedia'
        NUM_URLS = 7
        NUM_PAGES = 7

        all_urls = get_urls(QUERY, NUM_URLS)
        web_sentences = []
        web_sentences.extend(get_sentences_web(NUM_PAGES, all_urls))
        web_sentences = [unidecode.unidecode(clean_text(sent)) for sent in web_sentences]

        if len(web_sentences) == 0:
            print(split, title)
        
        random.shuffle(web_sentences)
        
        out.append(web_sentences)
        
    c[split] = out
    
with open('./google_re_corpora.pkl', 'wb') as handle:
    pickle.dump(c, handle, protocol=pickle.HIGHEST_PROTOCOL)

