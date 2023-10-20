# Text Fact Transfer
ModQGA Framework described in the Text Fact Transfer Paper

## Overview

This repository corresponds to code for the paper "Text Fact Transfer".

## Setup

Python 3.8.12 and pip 21.3.1 were used when running this module. A list of requirements can be found in `requirements.txt`, which can be installed through the following command:
```
pip install -r requirements.txt 
```
## Functional Design

The `model` folder contains the scripts necessary for training the components of ModQGA and running the model. The folder is organized as follows:
* `model/ModQGA.py`: Inference script for running ModQGA model and extracting transferred entities
* `model/seq2seq.py`: Replacement model that is run after extracting transferred entities for 0-shot ModQGA
* `model/zero_shot.py`: LED model that is run after extracting transferred entities for ModQGA-Sup
* `model/Question Answering.ipynb`: Jupyter notebook to train the specificity-aware question answering model.
* `model/Question Generation.ipynb`: Jupyter notebook to train the end-to-end question generation model.

The `evaluate` folder contains the scripts to evaluate the generated text. The folder is organized as follows:
* `evaluate/nli_eval.py`: Run the "Natural Language Inference - Entailment" metric
* `evaluate/output_similarity_eval.py`: Run the output similarity metrics, including ROUGE, BLEU, METEOR, BERTScore, and Halluc
