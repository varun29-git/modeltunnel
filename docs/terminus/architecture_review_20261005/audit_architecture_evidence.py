"""Offline architecture arithmetic and ternary-scale diagnostics; no training.

The optional checkpoint must be the verified dense 10B parent. Quantization
error is a weight-space diagnostic, not a prediction of language-model loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

DENSE_SHA = "3d116e88048c93a0403ab2920b1680b066a9f29cbc07fa8ae1c2c67f2144684b"


def shape(width, ffn, heads, kv_heads, *, layers=25, qk_gain=True):
    head = width // heads
    projections = layers * (2 * width * width + 2 * width * kv_heads * head + 3 * width * ffn)
    embedding = 65536 * width
    main_norm = (2 * layers + 1) * width
    qk_norm = 2 * head * layers if qk_gain else 0  # head-dim gain shared across heads
    total = projections + embedding + main_norm + qk_norm
    scale_count = projections // (128 * 128)
    return dict(width=width, ffn=ffn, query_heads=heads, kv_heads=kv_heads, head_dim=head,
                layers=layers, projection_parameters=projections, embedding_parameters=embedding,
                main_norm_parameters=main_norm, qk_norm_parameters=qk_norm,
                total_parameters=total, embedding_fraction=embedding/total,
                learned_ternary_block_scales=scale_count,
                total_with_learned_block_scales=total+scale_count,
                theoretical_export_bytes_2bit_codes_bf16_embedding=(projections//4 + embedding*2
                    + (main_norm+qk_norm+scale_count)*4),
                bf16_weight_bytes=2*total,
                row_group_export={str(g):dict(scale_count=projections//g,
                    fp16_scale_bits_per_projection_weight=16/g,
                    total_weight_bytes_2bit_codes_bf16_embedding_fp16_scales=(projections//4+embedding*2
                        +(main_norm+qk_norm)*4+(projections//g)*2)) for g in (32,64,128)})


def attention_cost(sequence, window=512, layers=25, globals_=5, kv_heads=4, head_dim=128):
    global_pairs=sequence*(sequence+1)//2
    local_pairs=sequence*window-window*(window-1)//2
    return dict(sequence=sequence, window=window,
                causal_attention_pair_ratio=(globals_*global_pairs+(layers-globals_)*local_pairs)/(layers*global_pairs),
                bf16_kv_bytes=2*2*kv_heads*head_dim*(globals_*sequence+(layers-globals_)*window),
                all_global_bf16_kv_bytes=2*2*kv_heads*head_dim*layers*sequence)


def quantizer_diagnostics(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=DENSE_SHA:raise ValueError('checkpoint SHA mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    totals={name:dict(count=0, original_sq=0., error_sq=0., effective_sq=0., zeros=0)
            for name in ('absmean','fixed_codes_centroid','alternating_centroid_8',
                         'row32_absmean','row64_absmean','row128_absmean')}
    for name,weight in payload['model'].items():
        if weight.ndim!=2 or not name.endswith(('.qkv_proj.weight','.o_proj.weight','.gate_up_proj.weight','.down_proj.weight')):continue
        rows,cols=weight.shape
        assert rows%128==0 and cols%128==0
        w=weight.float().reshape(rows//128,128,cols//128,128).permute(0,2,1,3).contiguous()
        a=w.abs().mean((-1,-2),keepdim=True).clamp_min(1e-6)
        original_sq=w.double().square().sum().item()
        codes=(w/a).round().clamp(-1,1)
        fixed_alpha=(w*codes).sum((-1,-2),keepdim=True)/codes.square().sum((-1,-2),keepdim=True).clamp_min(1)
        centroid=fixed_alpha.clamp_min(1e-6)
        for _ in range(8):
            alternate=(w/centroid).round().clamp(-1,1)
            centroid=((w*alternate).sum((-1,-2),keepdim=True)/alternate.square().sum((-1,-2),keepdim=True).clamp_min(1)).clamp_min(1e-6)
        # Evaluate the code assignment for the final scale, not the prior iteration.
        alternate=(w/centroid).round().clamp(-1,1)
        for key,effective in [('absmean',codes*a),('fixed_codes_centroid',codes*fixed_alpha),('alternating_centroid_8',alternate*centroid)]:
            item=totals[key];item['count']+=w.numel();item['original_sq']+=original_sq
            item['error_sq']+=(w.double()-effective.double()).square().sum().item()
            item['effective_sq']+=effective.double().square().sum().item()
            item['zeros']+=(effective==0).sum().item()
        for group in (32,64,128):
            row_w=weight.float().reshape(-1,group)
            alpha=row_w.abs().mean(-1,keepdim=True).clamp_min(1e-6)
            effective=(row_w/alpha).round().clamp(-1,1)*alpha
            item=totals[f'row{group}_absmean'];item['count']+=row_w.numel();item['original_sq']+=original_sq
            item['error_sq']+=(row_w.double()-effective.double()).square().sum().item()
            item['effective_sq']+=effective.double().square().sum().item()
            item['zeros']+=(effective==0).sum().item()
    return {key:dict(parameters=v['count'],relative_rms_error=math.sqrt(v['error_sq']/v['original_sq']),
                     effective_rms_ratio=math.sqrt(v['effective_sq']/v['original_sq']),zero_fraction=v['zeros']/v['count'])
            for key,v in totals.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dense-checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    result={'scope':'proposed shape arithmetic; offline quantization only; no task or training quality measured',
            'family':{str(d):shape(d,7*d//2,d//128,d//512) for d in (512,1024,1536,2048)},
            'optional_depth_comparator':shape(2048,5632,16,4,layers=30),
            'historical_150m':shape(640,1664,5,1,qk_gain=False),
            'historical_2304_geometry_without_subln':shape(2304,5760,18,6,qk_gain=False),
            'attention':{str(s):attention_cost(s) for s in (4096,32768)},
            'initial_vs_nominal_peak_muon_update':{
                'caveat':'nominal approximate update RMS at peak divided by initialization RMS; not a measured update ratio',
                'base_peak_lr':.01,'nominal_update_rms':.002,
                'input_weight_initial_rms':1/math.sqrt(640),
                'o_initial_rms':.14078571632817444/math.sqrt(640),
                'down_initial_rms':.14078571632817444/math.sqrt(1664)}}
    if args.dense_checkpoint:
        result['dense_checkpoint_sha256']=DENSE_SHA
        result['offline_ternary_scale']=quantizer_diagnostics(args.dense_checkpoint)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
