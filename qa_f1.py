import nltk
from rouge_score import rouge_scorer
import evaluate
from datasets import load_dataset
from thefuzz import fuzz
from multiprocessing import Pool

import json

# Metric
metric = evaluate.load("f1")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(preds,labels):
    if isinstance(preds, tuple):
        preds = preds[0]

    raw_dataset = load_dataset("/home/jqcao/huggingface/narrativeqa",split="test")

    idx = 0
    pred_bin = []
    label_bin = []

    for sample in enumerate(raw_dataset):
        answer_num = len(sample["answers"])
        pred_span = preds[idx,idx+answer_num]
        label_span = labels[idx,idx+answer_num]
        idx += answer_num
        
        for i,answer in enumerate(sample["answers"]):
            answer_text = answer["text"].strip().replace("\n", " ")
            print(label_span[i], answer_text)
            assert(label_span[i] == answer_text)
        
        result = False
        pred_label_pair = [(a,b) for a in pred_span for b in label_span]
        for a,b in pred_label_pair:
            if(fuzz.partial_token_set_ratio(a,b)>90):
                result = True
        if(result):
            pred_bin.append(1)
            label_bin.append(1)
        else:
            pred_bin.append(0)
            label_bin.append(1)
        
    assert(len(pred_bin)==len(raw_dataset))
    # use_stemmer=True to change words to their original form
    result = metric.compute(predictions=pred_bin, references=label_bin)
    return result

def filter_predict(predictions):
    filtered=[]
    for prediction in predictions:
        length=len(prediction)
        first_period = prediction.find(".")
        second_period = prediction.find(".",first_period+1)
        
        if(length<30 or first_period==-1):
            filtered.append(prediction)
        elif(first_period!=-1 and second_period==-1):
            filtered.append(prediction[:first_period+1])
        elif(first_period!=-1 and second_period!=-1):
            if(prediction[:first_period]==prediction[first_period+1:second_period]):
                filtered.append(prediction[:first_period+1])
            else:
                filtered.append(prediction[:second_period+1])
        else:
            print(f"ERROR for {prediction}")
        
    return filtered

file = "./outputs/export/narrativeqa/multi_answer_3epoch/generated_predictions.jsonl"
prediction=[]
label=[]
with open(file,"r") as file:
    for line in file:
        raw_data = json.loads(line)
        prediction.append(raw_data["predict"])
        label.append(raw_data["label"])
    
prediction = filter_predict(prediction)
print(compute_metrics(prediction,label))
