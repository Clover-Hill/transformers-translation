import nltk
import evaluate
from multiprocessing import Pool
from tqdm import tqdm

import jsonlines

# Metric
# ---------------------------------------
# rouge result: {'rouge1': 59.6417, 'rouge2': 33.1355, 'rougeL': 55.5787, 'rougeLsum': 55.6493}
# ---------------------------------------
# bleu result: {'bleu': 0.25700355975539846, 'precisions': [0.5972557784984303, 0.32406769588178885, 0.19353697703702655, 0.11797369124328176], 'brevity_penalty': 0.9967880734798, 'length_ratio': 0.9967932206882341, 'translation_length': 77399, 'reference_length': 77648}
# ---------------------------------------
metric = evaluate.load("bleu")

def postprocess_text(preds, labels):
    if isinstance(preds, str):
        preds = [preds]
        labels = [labels]
    
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(preds,labels):
    # Some simple post-processing
    post_preds, post_labels = postprocess_text(preds, labels)

    # use_stemmer=True to change words to their original form
    result = metric.compute(predictions=post_preds, references=post_labels)
    # result = {k: round(v * 100, 4) for k ,v in result.items()}
    return result

def sort_metrics(output_file, metric_name):
    eval_metric = []
    for line in tqdm(data, desc="Calculation Metrics"):
        post_pred, post_label = postprocess_text(line["pred"],line["label"])
        
        result = metric.compute(predictions=post_pred, references=post_label)
        eval_metric.append(round(result[metric_name]*100,4))
    
    merge_result = list(zip(data,eval_metric))
    merge_result = sorted(merge_result, key=lambda x:x[1])
    
    with jsonlines.open(output_file,"w") as file:
        for line in tqdm(merge_result, desc="Writing Result"):
            cur = line[0]
            cur[metric_name] = line[1]
            file.write(cur)

file = "./generated_result.jsonl"
prediction=[]
label=[]
data=[]
with jsonlines.open(file,"r") as file:
    for id,line in tqdm(enumerate(file), unit='lines'):
        new_line = {"id":id, **line}
        data.append(new_line)
        prediction.append(line["pred"])
        label.append(line["label"])
    
sort_metrics("./bleu_result.jsonl","bleu")
# print(compute_metrics(prediction,label))