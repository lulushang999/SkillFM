"""Generate one skill per public task with exactly one Flow evaluation per split."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import torch
from skillflow.io import load_config, read_jsonl, sha256
from skillflow.flow.model import VelocityField, sample_meanflow_prior, sample_meanflow_endpoint
from skillflow.codec.classic.model import FrozenQueryEncoder, SkillAutoencoderCodec
from skillflow.skills.executable import parse_executable_skill


def public_conditions(rows, dataset):
    """Project the input onto public task fields; answers never reach the model."""
    contexts=[]
    for row in rows:
        goal=row.get('canonical_goal') or row.get('goal') or row.get('query') or row.get('question')
        task_type=row.get('task_type')
        if not task_type and dataset in ('nq','multihop'): task_type='direct_retrieval'
        if not task_type and dataset=='webshop':
            from skillflow.skills.webshop_taxonomy import classify_webshop_task_type
            task_type=classify_webshop_task_type(goal)
        if not isinstance(goal,str) or not goal.strip() or not task_type:
            raise ValueError(f"Missing public goal or task_type: {row['task_id']}")
        contexts.append({'task_id':str(row['task_id']),'task_type':str(task_type),'canonical_goal':goal})
    return contexts


def generate(config, codec_path, checkpoint, tasks, output, device='cuda:0'):
    cfg=load_config(config); c=cfg['codec']; device=torch.device(device)
    contexts=public_conditions(read_jsonl(tasks),cfg['dataset'])
    out=Path(output); out.parent.mkdir(parents=True,exist_ok=True)
    # Reserve the output before any model invocation; never retry an existing run.
    with out.open('x',encoding='utf-8') as f:
        ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if ckpt.get('nfe') != 1: raise ValueError('Expected a release NFE=1 checkpoint')
        if ckpt['codec_sha256'] != sha256(Path(codec_path)/'vector_to_prefix.pt'):
            raise ValueError('Flow and Codec do not belong to the same training run')
        if ckpt['config']['dataset'] != cfg['dataset']:
            raise ValueError('Flow checkpoint belongs to another dataset')
        contract=torch.load(Path(codec_path)/'vector_to_prefix.pt',map_location='cpu',weights_only=False)
        if contract['encoder_query_prompt'] != c['encoder_query_prompt']:
            raise ValueError('Query prompt differs from the frozen Codec')
        query_fingerprint=contract['query_encoder_source_fingerprint']
        query_pooling=contract['query_encoder_pooling']
        del contract
        torch.set_float32_matmul_precision(cfg['flow']['float32_matmul_precision'])
        query=FrozenQueryEncoder(c['query_encoder_model'],device=device,max_length=c['max_length'],
            query_prompt=c['encoder_query_prompt'],backbone_dtype=c['backbone_dtype'],
            attention_implementation=c['attention_implementation'],
            expected_source_fingerprint=query_fingerprint,pooling_profile=query_pooling)
        q=query.encode([r['canonical_goal'] for r in contexts],task_types=[r['task_type'] for r in contexts],
            batch_size=cfg['models']['skill_encode_batch_size'])
        del query; gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
        model=VelocityField(**ckpt['model_config']).to(device)
        model.load_state_dict(ckpt['state_dict'])
        generator=torch.Generator(device=device).manual_seed(int(cfg['seed']))
        epsilon=sample_meanflow_prior(len(contexts),model.z_dim,device,generator=generator)
        # The sampler enforces NFE=1; decoding batches do not add Flow calls.
        latents=sample_meanflow_endpoint(model,epsilon,torch.from_numpy(q).to(device),sampling_steps=1).cpu()
        del model,epsilon,q,ckpt; gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
        codec=SkillAutoencoderCodec(codec_path,device=device,max_length=c['max_length'],
            backbone_dtype=c['backbone_dtype'],attention_implementation=c['attention_implementation'])
        codec.offload_encoder()
        results=codec.decode_results(latents,contexts=contexts,
            batch_size=cfg['models']['skill_decode_batch_size'],
            max_new_tokens=cfg['models']['skill_decode_max_new_tokens'])
        if len(results)!=len(contexts): raise RuntimeError('Decoder changed output cardinality')
        for context,result in zip(contexts,results):
            skill=parse_executable_skill(result.text) if result.status=='ok' else None
            status=result.status if result.status!='ok' else ('ok' if skill else 'invalid_skill')
            row={**context,'skill':skill,'status':status,'nfe':1,'decode_attempts':1}
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
        f.flush()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','codec','checkpoint','tasks','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args(); generate(a.config,a.codec,a.checkpoint,a.tasks,a.output,a.device)

if __name__=='__main__': main()
