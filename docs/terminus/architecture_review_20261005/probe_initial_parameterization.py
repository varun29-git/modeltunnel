"""CPU-only initialization diagnostic, not training or language-model evaluation.

Applies the proposed block equations using existing reference modules. Random
token IDs expose tied-embedding self-logit correlation before any optimization.
"""
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from terminus.model import BitNetLlama, BitNetLlamaConfig, RMSNorm

torch.set_num_threads(4)
torch.manual_seed(1337)
d=512
config=BitNetLlamaConfig(vocab_size=65536,dim=d,n_layers=25,n_heads=4,n_kv_heads=1,ffn_dim=1792,
    attention_backend='sdpa_reference',use_subln=False,qk_norm_mode='none',fuse_projections=True,
    weight_mode='dense',residual_init_scale=1.,local_attention_window=512,global_attention_every=5)
model=BitNetLlama(config).eval()
for layer in model.layers:
    layer.attn.q_norm=RMSNorm(128)
    layer.attn.k_norm=RMSNorm(128)
with torch.inference_mode():
    ids=torch.randint(32,65536,(2,32))
    targets=torch.randint(32,65536,ids.shape)
    hidden=math.sqrt(d)*model.tok_embeddings(ids)
    input_rms=hidden.square().mean().sqrt().item()
    cos,sin=model.rotary(ids.shape[1],device=hidden.device,dtype=hidden.dtype)
    for layer in model.layers:
        hidden=hidden+layer.attn(layer.attn_norm(hidden),cos,sin)/math.sqrt(50)
        hidden=hidden+layer.ffn(layer.ffn_norm(hidden))/math.sqrt(50)
    raw=F.linear(model.norm(hidden),model.tok_embeddings.weight)
    rows=[]
    for label,tau in [('unscaled',1.),('c_logit_1',1/math.sqrt(d)),('c_logit_2',2/math.sqrt(d))]:
        z=raw*tau;logp=z.log_softmax(-1)
        rows.append(dict(name=label,multiplier=tau,random_target_nll=F.cross_entropy(z.reshape(-1,65536),targets.reshape(-1)).item(),
            mean_entropy=(-(logp.exp()*logp).sum(-1)).mean().item(),
            mean_current_token_logit=z.gather(-1,ids[...,None]).mean().item(),
            mean_current_token_probability=logp.exp().gather(-1,ids[...,None]).mean().item(),
            maximum_absolute_logit=z.abs().max().item()))
out=dict(scope='untrained width-512 reference model; FP32 arithmetic; random IDs; not real-data loss',
    seed=1337,shape=[2,32],parameters=sum(p.numel() for p in model.parameters()),
    input_residual_rms=input_rms,uniform_nll=math.log(65536),readout_comparisons=rows)
Path(__file__).with_name('initialization_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
