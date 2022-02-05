
import nltk  # Here to have a nice missing dependency error message early on
import six  # Here to have a nice missing dependency error message early on
from generator import *
from datasets import load_metric

text_num = 19
myrouge = load_metric("rouge")
preds = list()
references = list()
metric_only=True
if metric_only:
    result_file = open('generated_predictions.txt','r',encoding='utf-8').readlines()
    preds = result_file
    print(len(preds))
    for index in range(text_num):
        references.append(get_demo(index)['abstract'])
else:
    result_file = open('generate_results.txt','w')
    for index in range(text_num):
        textitem = get_demo(index)
        pred = generate(textitem['description']).strip()
        result_file.write("\n"+pred)
        reference = textitem['abstract']
        preds.append(pred)
        references.append(reference)

preds = [pred.strip() for pred in preds]
references = [reference.strip() for reference in references]
preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
references = ["\n".join(nltk.sent_tokenize(reference)) for reference in references]    
result = myrouge.compute(predictions=preds,references = references,use_stemmer=True)
result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
print(result)    
