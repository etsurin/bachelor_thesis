from tqdm import tqdm
import nltk  
from generator import *
from datasets import load_metric

compute_metric = load_metric("rouge")


def rouge(preds, references,rouge_types=None):
    '''
    preds and references can be both str or list[str]
    rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    '''
    if isinstance(preds,str):
        preds = ["\n".join(nltk.sent_tokenize(preds))]
        references = ["\n".join(nltk.sent_tokenize(references))]
    else:
        preds = [pred.strip() for pred in preds]
        references = [reference.strip() for reference in references]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        references = ["\n".join(nltk.sent_tokenize(reference)) for reference in references]
    result = compute_metric.compute(predictions=preds,references = references,rouge_types=rouge_types,use_stemmer=True)
    result = {key: round(value.mid.fmeasure * 100,2) for key, value in result.items()}
    return result

def run_and_test(test_only = True):
    preds = list()
    references = list()
    if test_only:
        result_file = open('./ami_fine_annobase.txt','r').readlines()
        datasetname='ami_test'
        data =processjson(datasetname)
        preds = result_file
        print(len(preds))
        for index in range(len(preds)):
            references.append(data[index]['abstract'])
    else:
        infolist = pickle.load(open('ami_info_test.pkl','rb'))
        filename = 'ami_testseg.txt'
        preds = merge_predictions(filename,infolist)
        # print(preds)
        data = processjson('ami_test')
        for index in range(len(data)):
            references.append(data[index]['abstract'])
    result = rouge(preds,references)
    return result

def main():
    print(run_and_test())


if __name__=='__main__':
    main()