"""Train the classic iMF with a frozen Codec and training-only candidate selection."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from skillflow.io import load_config, training_rows, sha256, write_json
from skillflow.flow.model import (VelocityField, project_unit, sample_meanflow_prior,
    sample_meanflow_times, improved_meanflow_boundary_target, sample_meanflow_endpoint)
from skillflow.flow.coverage_sampler import CoverageFirstDDPSampler
from skillflow.flow.selection import (_stable_group_key, _condition_set_score_components,
    _flow_learning_rate_for_step, fixed_subset, operator, assess_decode, aggregate, rank_key)


def split_rows(rows, flow):
    """Stratify complete public task conditions; never select from test data."""
    by_type = {}
    for row in rows:
        by_type.setdefault(row['task_type'], set()).add((row['task_type'], row['canonical_goal']))
    if flow['internal_validation_partition'] not in {
            'task_type_stratified_semantic_condition_holdout_v1', 'training_subset_probe_v1'}:
        raise ValueError('Unsupported Flow validation partition')
    selected = set()
    for conditions in by_type.values():
        ordered = sorted(conditions, key=lambda c: (_stable_group_key(flow['internal_validation_seed'], c), c))
        n = min(len(ordered)-1, max(1, round(len(ordered)*flow['internal_validation_ratio']))) if len(ordered)>1 else 0
        selected.update(ordered[:n])
    probe = [i for i,r in enumerate(rows) if (r['task_type'], r['canonical_goal']) in selected]
    if not probe: raise ValueError('Training data cannot form a semantic probe')
    independent = flow['internal_validation_partition'] == 'task_type_stratified_semantic_condition_holdout_v1'
    probe_set = set(probe)
    train = [i for i in range(len(rows)) if not independent or i not in probe_set]
    return train, probe, independent


def hierarchy_weights(rows):
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row['task_type'], {}).setdefault(row['canonical_goal'], []).append(i)
    weights = np.zeros(len(rows), dtype=np.float64)
    for conditions in groups.values():
        for indices in conditions.values():
            weights[indices] = 1.0 / (len(groups)*len(conditions)*len(indices))
    weights /= weights.sum()
    return weights


def make_model(flow, z_dim, q_dim, device):
    return VelocityField(z_dim=z_dim, q_dim=q_dim, hidden_dim=flow['hidden_dim'],
        depth=flow['depth'], time_dim=flow['time_dim'], context_dim=flow['context_dim'],
        attention_heads=flow['num_heads'], latent_token_size=flow['latent_token_size'],
        mlp_ratio=flow['mlp_ratio'], architecture=flow['architecture'],
        output_parameterization=flow['output_parameterization'],
        condition_injection=flow['condition_injection']).to(device)


def stage_at(flow, step):
    for stage in flow.get('batch_schedule', []):
        if int(stage['start_step']) <= step <= int(stage['end_step']): return stage
    return {'start_step': 1, 'end_step': flow['training_steps'],
            'global_batch_size': flow['global_batch_size'], 'fused_optimizer': flow['fused_optimizer']}


def select_candidates(cfg, candidates, probe_rows, z, q, codec_path, output, device):
    flow = cfg['flow']
    if not candidates:
        raise ValueError('No Flow checkpoint candidates')
    generator = torch.Generator(device=device).manual_seed(flow['internal_validation_seed'])
    prior = sample_meanflow_prior(len(probe_rows), z.shape[1], device, generator=generator)
    dataset = SimpleNamespace(terminal_latents=z.cpu().numpy(), size=len(probe_rows),
        task_types=[r['task_type'] for r in probe_rows],
        canonical_goals=[r['canonical_goal'] for r in probe_rows], condition_mode='task_type_goal_v1')
    records, subsets = [], []
    decoded_selection = cfg['dataset'] == 'multihop'
    subset = fixed_subset([r['task_id'] for r in probe_rows], count=min(64,len(probe_rows)),
                         strata=[operator(r['canonical_goal']) for r in probe_rows]) if decoded_selection else []
    for path in candidates:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = VelocityField(**ckpt['model_config']).to(device)
        model.load_state_dict(ckpt['state_dict'])
        # One field evaluation per complete probe, just as in final generation.
        endpoints = sample_meanflow_endpoint(model, prior, q, sampling_steps=1)
        score = _condition_set_score_components(endpoints.cpu().numpy(), dataset)['chamfer_mse']
        records.append({'step': ckpt['step'], 'checkpoint': path.name,
                        'geodesic_mse': score, 'nfe': 1})
        if subset: subsets.append(endpoints[subset].cpu())
        del model, endpoints, ckpt
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if decoded_selection:
        from skillflow.codec.classic.model import SkillAutoencoderCodec
        c = cfg['codec']
        codec = SkillAutoencoderCodec(codec_path, device=device, max_length=c['max_length'],
            backbone_dtype=c['backbone_dtype'], attention_implementation=c['attention_implementation'])
        codec.offload_encoder()
        for record, latent in zip(records, subsets):
            contexts = [probe_rows[i] for i in subset]
            results = codec.decode_results(latent, contexts=contexts,
                batch_size=cfg['models']['skill_decode_batch_size'],
                max_new_tokens=cfg['models']['skill_decode_max_new_tokens'])
            if len(results) != len(subset):
                raise RuntimeError('Candidate decoding returned an incomplete probe')
            record['decoded'] = aggregate([assess_decode(r.text, r.status,
                (probe_rows[i]['executable_skill'] if isinstance(probe_rows[i]['executable_skill'], dict)
                 else json.loads(probe_rows[i]['executable_skill']))) for r,i in zip(results,subset)])
        winner = min(records, key=rank_key)
    else:
        winner = min(records, key=lambda r: (r['geodesic_mse'],r['step']))
    import shutil
    shutil.copyfile(output / winner['checkpoint'], output / 'flow_best.pt')
    write_json(output / 'selection.json', {'selected_step':winner['step'],
        'nfe':1, 'candidates':records, 'test_data_used':False,
        'partition':flow['internal_validation_partition'],
        'criterion':'schema_eos_then_rule_feature_proxy_then_geometry' if decoded_selection else 'condition_symmetric_chamfer_geodesic_mse'})


def train(config, pairs, codec_path, output):
    cfg = load_config(config); rows = training_rows(cfg); flow = cfg['flow']
    metadata = json.loads((Path(pairs)/'metadata.json').read_text())
    if metadata['skills_sha256'] != sha256(cfg['data']['skills']): raise ValueError('Training data changed after encoding')
    if metadata['codec_sha256'] != sha256(Path(codec_path)/'vector_to_prefix.pt'): raise ValueError('Codec changed after encoding')
    if metadata['task_ids'] != [r['task_id'] for r in rows]: raise ValueError('Latent row order mismatch')
    if metadata.get('query_encoder_prompt') != cfg['codec']['encoder_query_prompt']:
        raise ValueError('Query encoder prompt changed after encoding')
    if metadata.get('query_encoder_model') != cfg['codec']['query_encoder_model']:
        raise ValueError('Query encoder source changed after encoding')
    rank = int(os.environ.get('RANK',0)); world = int(os.environ.get('WORLD_SIZE',1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK',0))) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda': torch.cuda.set_device(device)
    if world > 1: dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    out = Path(output)
    if rank == 0: out.mkdir(parents=True, exist_ok=False)
    if world > 1: dist.barrier()
    torch.set_float32_matmul_precision(flow['float32_matmul_precision'])
    torch.manual_seed(cfg['seed'])
    data = np.load(Path(pairs)/'pairs.npz', allow_pickle=False)
    for key in ('z', 'q'):
        value = data[key]
        if value.ndim != 2 or value.shape[0] != len(rows) or value.shape[1] == 0:
            raise ValueError('Latent shape/cardinality mismatch')
        if not np.isfinite(value).all() or np.any(np.linalg.norm(value, axis=1) <= 1e-8):
            raise ValueError('Encoded latent contains nonfinite or zero-norm rows')
    train_idx, probe_idx, independent = split_rows(rows, flow)
    all_z = torch.from_numpy(data['z']).to(device).float()
    all_q = torch.from_numpy(data['q']).to(device).float()
    z = project_unit(all_z[train_idx]); q = project_unit(all_q[train_idx])
    weights = hierarchy_weights([rows[i] for i in train_idx])
    if rank == 0: write_json(out/'split.json', {'gradient_task_ids':[rows[i]['task_id'] for i in train_idx],
        'probe_task_ids':[rows[i]['task_id'] for i in probe_idx], 'independent_flow_holdout':independent})
    model = make_model(flow,z.shape[1],q.shape[1],device)
    wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type=='cuda' else None) if world>1 else model
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    candidates = []; current_stage = None
    try:
        for step in range(1, int(flow['training_steps'])+1):
            stage = stage_at(flow, step)
            if current_stage != stage:
                if stage['global_batch_size'] % world: raise ValueError('Global Flow batch must be divisible by world size')
                batch = stage['global_batch_size']//world
                if stage.get('restart_from_ema'):
                    model.load_state_dict(ema.state_dict())
                optimizer = torch.optim.Adam(model.parameters(), lr=flow['learning_rate'],
                    betas=(flow['adam_beta1'],flow['adam_beta2']), weight_decay=flow['weight_decay'],
                    fused=bool(stage['fused_optimizer']) and device.type=='cuda')
                coverage = CoverageFirstDDPSampler(pair_count=len(z),local_batch_size=batch,
                    world_size=world,seed=cfg['seed'])
                current_stage = stage
            lr = _flow_learning_rate_for_step(step, training_steps=flow['training_steps'],
                peak_learning_rate=flow['learning_rate'],minimum_learning_rate=flow['minimum_learning_rate'],
                warmup_steps=flow['learning_rate_warmup_steps'],schedule=flow['learning_rate_schedule'])
            for group in optimizer.param_groups: group['lr']=lr
            rng = np.random.default_rng(cfg['seed'] * 1_000_003 + rank * 10_007 + step - 1)
            if step-1 < coverage.coverage_steps:
                item = coverage.batch(step=step-1,rank=rank)
                indices = item.indices.copy()
                padding = ~item.valid_mask
                if padding.any():
                    indices[padding] = rng.choice(len(z), size=int(padding.sum()), replace=True, p=weights)
                valid_count = min(coverage.global_batch_size,len(z)-(step-1)*coverage.global_batch_size)
                example_weights = torch.as_tensor(item.valid_mask,device=device,dtype=torch.float32)*world/valid_count
            else:
                indices = rng.choice(len(z),size=batch,replace=True,p=weights)
                example_weights = torch.full((batch,),1.0/batch,device=device)
            index = torch.as_tensor(indices,device=device)
            target = z[index]*math.sqrt(z.shape[1]); query=q[index]
            generator = torch.Generator(device=device).manual_seed(cfg['seed']*1000003+rank*10007+(step-1)*97+17)
            epsilon=sample_meanflow_prior(batch,z.shape[1],device,generator=generator)
            r,t=sample_meanflow_times(batch,device=device,logit_normal_mean=flow['time_mean'],
                logit_normal_std=flow['time_std'],offdiag_probability=flow['off_diagonal_probability'],generator=generator)
            path=(1-t[:,None])*target+t[:,None]*epsilon
            target_v,_,_=improved_meanflow_boundary_target(model,path,r,t,query,conditional_velocity=epsilon-target)
            prediction=wrapped(path,r,t,query)
            squared=(prediction-target_v).float().square().sum(-1)
            loss=(squared/(squared.detach()+flow['adaptive_weight_eps']).pow(flow['adaptive_weight_power'])*example_weights).sum()
            finite=torch.tensor(int(torch.isfinite(loss).item()),device=device)
            if world>1: dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if not finite.item(): raise FloatingPointError(f'Nonfinite Flow loss at step {step}')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad_ok=torch.tensor(int(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())),device=device)
            if world>1: dist.all_reduce(grad_ok,op=dist.ReduceOp.MIN)
            if not grad_ok.item(): raise FloatingPointError(f'Nonfinite Flow gradient at step {step}')
            optimizer.step()
            with torch.no_grad():
                for e,p in zip(ema.parameters(),model.parameters()): e.lerp_(p,1-flow['ema_decay'])
                for e,p in zip(ema.buffers(),model.buffers()): e.copy_(p)
            if step%flow['log_every']==0 or step==1:
                value=loss.detach().clone()
                if world>1: dist.all_reduce(value); value/=world
                if rank==0: print(f'step={step} loss={value.item():.6f} lr={lr:.8g}',flush=True)
            if step%cfg['checkpoint']['eval_every_steps']==0 or step==flow['training_steps']:
                p=out/f'flow_step_{step:06d}.pt'; candidates.append(p)
                if rank==0:
                    torch.save({'state_dict':{k:v.cpu() for k,v in ema.state_dict().items()},
                        'model_config':model.config(),'step':step,'codec_id':metadata['codec_id'],
                        'codec_sha256':metadata['codec_sha256'],'config':cfg,'nfe':1},p)
                    tmp=out/'resume.tmp'
                    torch.save({'model':model.state_dict(),'ema':ema.state_dict(),
                        'optimizer':optimizer.state_dict(),'step':step,'config':cfg},tmp)
                    tmp.replace(out/'resume.pt')
                if world>1: dist.barrier()
        # Release DDP resources before the single-process candidate decoder.
        if world>1: dist.barrier(); dist.destroy_process_group()
        del wrapped, model, ema, optimizer
        gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
        if rank==0:
            select_candidates(cfg,candidates,[rows[i] for i in probe_idx],all_z[probe_idx],
                all_q[probe_idx],codec_path,out,device)
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','pairs','codec','output'): p.add_argument('--'+name,required=True)
    a=p.parse_args(); train(a.config,a.pairs,a.codec,a.output)

if __name__=='__main__': main()
