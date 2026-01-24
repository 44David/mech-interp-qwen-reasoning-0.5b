from transformer_lens import HookedTransformer
import transformer_lens.utils as utils
import circuitsvis as cv
from transformers import AutoModelForCausalLM, AutoTokenizer
import plotly.express as px
import torch
from functools import partial
from jaxtyping import Float
import tqdm as tqdm
from transformer_lens.hook_points import (
    HookPoint,
) 
import einops
import numpy as np
import json

device = "cpu" # running on my laptop, use cuda if you have a gpu
model_path = "44David/qwen-0.5b-reasoning-v2" # my model uploaded to hf

hf_model = AutoModelForCausalLM.from_pretrained(model_path,).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    hf_model=hf_model,
    tokenizer=tokenizer,
    device=device,
    center_writing_weights=False, # because we aren't using layernorm
    center_unembed=False,
)

def find_decision_layer(model, examples):    
    results = []
    
    for prompt in examples:
        logits, cache = model.run_with_cache(prompt)
        
        decision_pos = len(prompt.split()) - 1  
        
        think_token_id = model.tokenizer.encode('<t')[0] # "<t" since we're looking for <think> tags
        
        layer_logits = []
        for layer in range(model.cfg.n_layers):
            resid = cache["resid_post", layer][0, decision_pos]
            
            layer_logit = model.unembed(model.ln_final(resid))
            think_score = layer_logit[think_token_id].item()
            layer_logits.append(think_score)
            
        results.append(layer_logits)
    
    return results


with open('interp-dataset.jsonl', 'r') as f:
        data = json.load(f)

reasoning_task = [data['reasoning'][0]['prompt']]
factual_task = [data['factual'][0]['prompt']]

reasoning_scores = (find_decision_layer(model, reasoning_task))
factual_scores = (find_decision_layer(model, factual_task))

mean_reason_score = np.mean(reasoning_scores, axis=0)
mean_fact_score = np.mean(factual_scores, axis=0) 
 
divergence = mean_reason_score - mean_fact_score


print("layer divergence:\n")
for i, div in enumerate(divergence):
    print(f"layer {i:2d}: {div}")

