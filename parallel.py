from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import torch
from test_text import *
device='cuda'
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
device_map = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8,9,10,11],
                          1: [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
model.parallelize(device_map)                          
input=tokenizer(text_02,max_length=1024,return_tensors='pt')
input=torch.tensor(input['input_ids']).to(device)
summary_ids = model.generate(input, num_beams=10, max_length=200, early_stopping=True).cpu()
summary=[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
summary=summary[0]

print(summary)
