from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from divide_text import *
import torch
device='cuda:0' if torch.cuda.is_available() else 'cpu'

def get_model(model_name):
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    return model, tokenizer

model,tokenizer = get_model('./seg')

def generate(text):    
    if isinstance(text,list):
        summary=''
        for snippet in text:
            tmp_summary = generate(snippet)
            summary+=tmp_summary
        return summary
    input=tokenizer(text,max_length=1024,return_tensors='pt')
    input=torch.tensor(input['input_ids']).to(device)
    summary_ids = model.generate(input, num_beams=4, max_length=300, early_stopping=True).cpu()
    summary=[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    summary=summary[0]
    return summary

def generate_tokenlist(tokenlist):
    summary=''
    for tokens in tokenlist:
        input=torch.tensor([tokens]).to(device)
        summary_ids = model.generate(input, num_beams=4, max_length=200, early_stopping=True).cpu()
        output=[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        tmp_summary=output[0]
        summary+=tmp_summary
    return summary
    
def ami_test():
    data=processjson('ami_test')
    for item in data:
        summ = generate(item['description'])
        print(summ)

def main():
    ami_test()

if __name__ == 'main':
    main()
# print(generate(text_03))


