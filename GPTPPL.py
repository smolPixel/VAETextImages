from transformers import GPT2LMHeadModel, GPT2TokenizerFast

#https://huggingface.co/docs/transformers/perplexity
device = "cuda"
model_id = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
from datasets import load_dataset

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# pritest[:5]['text'])
# fds
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

import torch
from tqdm import tqdm

max_length = model.config.n_positions
stride = 512

bs=32
for i in tqdm(range(0, len(test['text']), bs)):
    text=test['text'][i:i+bs]
    encodings=tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids=encodings['input_ids']
    target_ids=input_ids.clone()
    outputs = model(input_ids, labels=target_ids)
    neg_log_likelihood=outputs[0]
    print(neg_log_likelihood)
    gfsd
fds


nlls = []
nll_maison=[]
for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1))
    trg_len = end_loc - i  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        print(outputs[1])
        print(outputs[1].shape)
        fds
        neg_log_likelihood = outputs[0] * trg_len

    nlls.append(neg_log_likelihood)

print(nlls)
print(len(nlls))
print(end_loc)
print(torch.stack(nlls).sum())
ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
print(ppl)