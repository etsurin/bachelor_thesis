from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np

device = 'cuda'
model_id = 'gpt2-medium'
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_id)

def get_loss(input,target,encoded=True):
#     input=input.to(device)
    input=torch.tensor(input).to(device)
    target=torch.tensor(target).to(device)
    #print(final_input,target)
    output=model(input)
    print(output.logits)
    output = output.logits[-1:]
    print(output)
    loss_fct = torch.nn.CrossEntropyLoss()
    #print(output.view(-1, output.size(-1)).shape, target.shape)
    loss = loss_fct(output.view(-1, output.size(-1)), target)
    return loss.cpu().detach()

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
text='I love wearing high'
target=' heels'
text1=tokenizer.encode(text)
tmp=list()
for i in range(100):
    tmp.append(0)
text2=tmp+text1
target=tokenizer.encode(target)
print(get_loss(text1,target))
print(get_loss(text2,target))