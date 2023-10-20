# Text Fact Transfer

## Overview

This repository corresponds to code for the EMNLP 2023 paper "Text Fact Transfer".

## Setup

Python 3.8.12 and pip 21.3.1 were used when running this module. A list of requirements can be found in `requirements.txt`, which can be installed through the following command:
```
pip install -r requirements.txt 
```
## Functional Design

The `data` folder contains the data needed to train the specificity-aware QA model, as well as the text fact transfer datasets. The t-rex dataset already has the corpora as part of the dataset, but the other corpora need to be collected. To collect the corpora for the expository text generation datasets (U.S. News and Medline), you can follow the instructions [here](https://github.com/nbalepur/expository-text-generation/). The code to obtain the corpora for the Google dataset can be found at `/data/scrape_web.py` 

The `model` folder contains the scripts necessary for training the components of ModQGA and running the model. The code should be run in the following order:

* `model/Question Answering.ipynb`: Jupyter notebook to train the specificity-aware question answering model
* `model/Question Generation.ipynb`: Jupyter notebook to train the end-to-end question generation model
* `model/ModQGA.py`: After the question answering and generation models of ModQGA have been run, we can run this inference script which extracts transferred entities
* `model/zero_shot.py`: After running `ModQGA.py`, this code will generate text in a 0-shot fashion
* `model/seq2seq.py`: After running `ModQGA.py`, this code will generate text in a supervised manner by training the LED language model

The `evaluate` folder contains the scripts to evaluate the generated text. The folder is organized as follows:
* `evaluate/nli_eval.py`: Run the "Natural Language Inference - Entailment" metric
* `evaluate/output_similarity_eval.py`: Run the output similarity metrics
FactCC can be calculated by following the instructions [here](https://github.com/salesforce/factCC).
