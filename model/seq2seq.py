import datasets
import copy
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

out_dir = '' # output directory
ds_dir = '' # dataset directory
modqga_data_dir = '' # keyphrases from modqga
res_name = '' # results name
input_name, output_name = 'corpus', 'output' # input/output column
max_output_length = 64 # maximum output length in seq2seq generation

model_name = "allenai/led-base-16384"
use_val = True

import pickle
with open(modqga_data_dir, 'rb') as handle:
    qa_data = pickle.load(handle)

data = datasets.load_dataset(ds_dir)
train_dataset, val_dataset, test_dataset = data["train"], data["val"], data["test"]


mod_qa_input = {key: [] for key in qa_data.keys()}
for key, val in qa_data.items():
    for _, _, all_questions, answers in val:
        
        input_text = ""
        for i in range(len(answers)):
            a = answers[i]
            input_text += f"<|answer|> {a} "    
        mod_qa_input[key].append(input_text)
        
train_dataset = train_dataset.add_column('qa', mod_qa_input['train'])
val_dataset = val_dataset.add_column('qa', mod_qa_input['val'])
test_dataset = test_dataset.add_column('qa', mod_qa_input['test'])

val_dataset_copy = copy.deepcopy(val_dataset)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
from transformers import AutoModelForSeq2SeqLM
led = AutoModelForSeq2SeqLM.from_pretrained(model_name)


max_input_length = 16384
batch_size = 1

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(
        ["<|title|> " + batch['title'][i] + " " + batch['qa'][i] + " <|template|> " + batch['template'][i] + " <|title2|> " + batch['template_title'][i] + " <|context|> " + ' '.join(batch[input_name][i]) for i in range(len(batch[input_name]))],
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
    )
    outputs = tokenizer(
        [output for output in batch[output_name]],
        padding="max_length",
        truncation=True,
        max_length=max_output_length,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch

train_dataset = train_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    #remove_columns=['aspect', 'title', 'general_web_sentences_with_wiki', 'history_web_sentences_no_wiki', 'history_web_sentences_with_wiki', 'output', 'output_aug'],
)


val_dataset = val_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    #remove_columns=['aspect', 'title', 'general_web_sentences_with_wiki', 'history_web_sentences_no_wiki', 'history_web_sentences_with_wiki', 'output', 'output_aug'],
)

test_dataset = test_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    #remove_columns=['aspect', 'title', 'general_web_sentences_with_wiki', 'history_web_sentences_no_wiki', 'history_web_sentences_with_wiki', 'output', 'output_aug'],
)

train_dataset.set_format(
   type="torch",
   columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)
test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

# set generate hyperparameters
# if 'led' in model_name:
#     led.config.num_beams = 4
#     led.config.max_length = 512
#     led.config.min_length = 32
#     led.config.length_penalty = 2.0
#     led.config.early_stopping = True
#     led.config.no_repeat_ngram_size = 3

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate

rouge = evaluate.load('rouge')

def compute_metrics(pred):

    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str,
        references=label_str,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
    )
    
    format_label = [[label.lower()] for label in label_str]
    pred_str = [x.lower() for x in pred_str]

    return {
        "R1": round(np.mean(rouge_output["rouge1"]), 4),
        "R2": round(np.mean(rouge_output["rouge2"]), 4),
        "RL": round(np.mean(rouge_output["rougeL"]), 4),
        "RLsum": round(np.mean(rouge_output["rougeLsum"]), 4)
    }

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    output_dir="./",
    logging_steps=250,
    num_train_epochs=20,
    eval_steps=5000,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    warmup_steps=1500,
    gradient_accumulation_steps=8
)

trainer = Seq2SeqTrainer(
    model=led,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

ret = trainer.predict(test_dataset, max_length = max_output_length)
ret_out = [''.join(tokenizer.batch_decode(ret.predictions[i], skip_special_tokens=True)) for i in range(len(test_dataset['title']))]
import pickle
with open(out_dir, 'wb') as handle:
    pickle.dump(ret_out, handle, protocol=pickle.HIGHEST_PROTOCOL)