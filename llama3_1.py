import torch
import math
import os
import torch.nn as nn
import inspect
from torch.nn import functional as F
from dataclasses import dataclass
import time
import torch.optim.adamw



# Load the state dict from the .pth file onto the specified device
state_dict = torch.load('/home/mairex/projects/test/llama3.1/Meta-Llama-3.1-8B-Instruct/consolidated.00.pth')

#Dtype??
for k, v in state_dict.items():
    print(k, v.shape, v.dtype)

# NO BIASES
bias_keys = [k for k in state_dict.keys() if 'bias' in k]
print("Bias terms found in the state dict:", bias_keys)

# NO WEITGHT SHARING
print(state_dict["output.weight"].data_ptr())
print(state_dict["tok_embeddings.weight"].data_ptr())
print((state_dict["output.weight"] == state_dict["tok_embeddings.weight"]).all())

@dataclass
class LlamaConfig:
    Llama8BI = {"block_size": 40000, "dim": 4096, "n_layers": 32, "n_heads": 32, "n_kv_heads": 8, "vocab_size": 128256, "ffn_dim_multiplier": 3.5, "multiple_of": 1024, "norm_eps": 1e-05, "rope_theta": 500000.0, "use_scaled_rope": True}


# TODO: research RoPE and implement in Attention, seems pretty difficult lol


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

class FeedForward(nn.Module):  #! SWIGLU
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config["dim"], config["dim"] * config["ffn_dim_multiplier"])
        self.w2 = nn.Linear(config["dim"] * config["ffn_dim_multiplier"], config["dim"] * config["ffn_dim_multiplier"])
        self.w3 = nn.Linear(config["dim"] * config["ffn_dim_multiplier"], config["dim"])

    

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = nn.LayerNorm(config['dim'], bias=False)
        self.ffn_norm = nn.LayerNorm(config['dim'], bias=False)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 

        self.tok_embeddings = nn.Embedding(config['vocab_size'], config['dim'])
        self.layers = nn.ModuleList([Block(config) for _ in range(config['n_layers'])])
        self.norm = nn.LayerNorm(config['dim'], bias=False)
        self.output = nn.Linear(config['vocab_size'], config['dim'], bias=False) # ! this is transposed in the state_dict 

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config['block_size'], f"cannot forward sequence of length {T}, block size is {self.config['block_size']}"
        rope = self.rope(self.config['block_size'])
        tok_embd = self.tok_embeddings(idx)
        x = rope + tok_embd
        for block in self.layers:
            x = block(x)
        x = self.norm(x)
        logits = self.output(x)
        
        return logits


model = Llama(LlamaConfig.Llama8BI)
