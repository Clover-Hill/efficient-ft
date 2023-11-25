import nltk
from rouge_score import rouge_scorer
import evaluate
from multiprocessing import Pool

import json

# Metric
metric = evaluate.load("rouge")

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

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(preds, labels)

    # use_stemmer=True to change words to their original form
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k ,v in result.items()}
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