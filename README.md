# Text Fact Transfer

## Overview

This repository corresponds to code for the EMNLP 2023 paper "Text Fact Transfer".

## Setup

Python 3.8.12 and pip 21.3.1 were used when running this module. A list of requirements can be found in `requirements.txt`, which can be installed through the following command:
```
pip install -r requirements.txt 
```
## Functional Design

The `data` folder contains the data needed to train the specificity-aware QA model, as well as the text fact transfer datasets

The `model` folder contains the scripts necessary for training the components of ModQGA and running the model. The folder is organized as follows:
* `model/ModQGA.py`: Inference script for running ModQGA model and extracting transferred entities
* `model/zero_shot.py`: After running `ModQGA.py`, this code will generate text in a 0-shot fashion
* `model/zero_shot.py`: After running `ModQGA.py`, this code will generate text in a supervised manner by training the LED language model
* `model/Question Answering.ipynb`: Jupyter notebook to train the specificity-aware question answering model
* `model/Question Generation.ipynb`: Jupyter notebook to train the end-to-end question generation model

The `evaluate` folder contains the scripts to evaluate the generated text. The folder is organized as follows:
* `evaluate/nli_eval.py`: Run the "Natural Language Inference - Entailment" metric
* `evaluate/output_similarity_eval.py`: Run the output similarity metrics
