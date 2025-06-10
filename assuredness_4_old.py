from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
import torch
import math
import json
import random
import torch.nn.functional as F
from torch import autocast
from scipy.stats import skew, kurtosis
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import os
huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
}
class Assuredness():
    def __init__(self):
        super().__init__()

        prim_model_name = 'opt-350m'


        self.prim_model = AutoModelForCausalLM.from_pretrained(prim_model_name, trust_remote_code=True,
                                                                torch_dtype=torch.bfloat16,
                                                                token=huggingface_config["TOKEN"])


        self.device_1 = torch.device("cuda:0" if torch.cuda.is_available() else "")

        self.prim_model.to(self.device_1)

        self.prim_model.eval()




        self.prim_tokenizer = AutoTokenizer.from_pretrained(prim_model_name)
        self.prim_tokenizer.pad_token = self.prim_tokenizer.eos_token
        self.prim_tokenizer.pad_token_id = self.prim_tokenizer.eos_token_id or self.prim_tokenizer.add_special_tokens(
            {'pad_token': '[PAD]'})




    def Prob_calculating(self, text, model, tokenizer, flag=0,prompt=""):
        if flag == 0:
            tokenized = tokenizer(text, return_tensors="pt", padding='max_length', return_token_type_ids=False,
                                  return_attention_mask=True, truncation = True, max_length = 512).to(self.device_1)
        elif flag == 2:
            tokenized = tokenizer(text, return_tensors="pt", padding='max_length', return_token_type_ids=False,
                                  return_attention_mask=True, truncation=True, max_length= 512).to(self.device_1)
        else:
            tokenized = tokenizer(text, return_tensors="pt", padding='max_length', return_token_type_ids=False,
                                  return_attention_mask=True, truncation = True, max_length = 512).to(self.device_2)

        # 确保 input_ids 是 Long 类型
        tokenized["input_ids"] = tokenized["input_ids"].long()
        labels = tokenized["input_ids"][:, 1:]
        attention_mask = tokenized["attention_mask"]

        #temperature =  self.temperature
        if flag == 0:
            with torch.no_grad():
                with autocast("cuda"):
                    outputs = model(**tokenized,output_hidden_states=True)
                    list_ = []
                    for i in range(0,25):
                        hidden = outputs.hidden_states[i]
                        if i != 24:
                            hidden = model.model.decoder.project_out(hidden)
                        logits_score = model.lm_head(hidden)[:, :-1]/ 1.0
                        list_.append(logits_score)
        elif flag == 2:
            with torch.no_grad():
                with autocast("cuda"):
                    logits_score = model(**tokenized,output_hidden_states=True).logits[:, :-1] / 1.0
        else:
            with torch.no_grad():
                with autocast("cuda"):
                    logits_score = model(**tokenized,output_hidden_states=True).logits[:, :-1] / 1.0

        # 计算概率
        #prob = self.get_likelihood(logits_score, labels, attention_mask).squeeze(0).item()

        return logits_score, labels, attention_mask

    def assuredness(self, text,model1, tokenizer1):
        logits1, labels1, attention_mask1 = self.Prob_calculating(text, model1,
                                                                  tokenizer1,flag=2)

        attention_mask = attention_mask1.to(torch.float16)[:, 1:]  # 忽略 [CLS] 或类似的起始标

        logits = logits1

        lprobs = torch.log_softmax(logits, dim=-1)  # (b,s,v)
        topk_logprobs, topk_indices = torch.topk(lprobs, k=5, dim=-1)  # shape:
        topk_probs = topk_logprobs.exp()
        probs = lprobs.exp()
        label_logprobs = lprobs.gather(dim=-1, index=labels1.unsqueeze(-1).long())  # (b,s)
        
        assuredness = ((((((topk_probs*(label_logprobs - topk_logprobs)).mean(dim=-1)) * attention_mask))).sum(
                dim=-1).sum(dim=-1)).item() / (attention_mask.sum(dim=-1).item())
        return assuredness


    def compute_crit(self,text):
        score = self.assuredness(text,self.prim_model,self.prim_tokenizer)

        return score