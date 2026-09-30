"""Encode training skills and public conditions with the frozen Codec/query encoder."""
from __future__ import annotations
import argparse
import gc
from pathlib import Path
import numpy as np
import torch
from skillflow.io import load_config, training_rows, sha256, write_json
from skillflow.codec.classic.model import SkillAutoencoderCodec, FrozenQueryEncoder
from skillflow.skills.executable import canonicalize_executable_skill


def prepare(config, codec_path, output, device='cuda:0'):
    cfg = load_config(config)
    rows = training_rows(cfg)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_float32_matmul_precision(cfg['flow']['float32_matmul_precision'])
    c = cfg['codec']
    runtime = SkillAutoencoderCodec(codec_path, device=device, load_decoder=False,
        max_length=c['max_length'], backbone_dtype=c['backbone_dtype'],
        attention_implementation=c['attention_implementation'])
    if runtime.encoder_query_prompt != c['encoder_query_prompt']:
        raise ValueError('Query encoder prompt differs from the frozen Codec contract')
    query_fingerprint = runtime.query_encoder_source_fingerprint
    query_pooling = runtime.query_encoder_pooling
    skills = [canonicalize_executable_skill(r['executable_skill']) for r in rows]
    unique = list(dict.fromkeys(skills))
    values = runtime.encode(unique, batch_size=cfg['models']['skill_encode_batch_size'])
    lookup = dict(zip(unique, values))
    z = np.stack([lookup[s] for s in skills])
    codec_id = runtime.codec_id
    del runtime, values, lookup
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    query = FrozenQueryEncoder(c['query_encoder_model'], device=device,
        max_length=c['max_length'], query_prompt=c['encoder_query_prompt'],
        backbone_dtype=c['backbone_dtype'], attention_implementation=c['attention_implementation'],
        expected_source_fingerprint=query_fingerprint, pooling_profile=query_pooling)
    q = query.encode([r['canonical_goal'] for r in rows],
        task_types=[r['task_type'] for r in rows], batch_size=cfg['models']['skill_encode_batch_size'])
    for name, value in (('skill', z), ('query', q)):
        if value.ndim != 2 or value.shape[0] != len(rows) or value.shape[1] == 0:
            raise ValueError(f'{name} encoder returned an invalid latent shape')
        if not np.isfinite(value).all() or np.any(np.linalg.norm(value, axis=1) <= 1e-8):
            raise ValueError(f'{name} encoder returned nonfinite or zero-norm rows')
    with (out / 'pairs.npz').open('xb') as f:
        np.savez(f, z=z, q=q)
    write_json(out / 'metadata.json', {'codec_id': codec_id,
        'codec_sha256': sha256(Path(codec_path) / 'vector_to_prefix.pt'),
        'skills_sha256': sha256(cfg['data']['skills']), 'count': len(rows),
        'task_ids': [r['task_id'] for r in rows],
        'query_encoder_model': c['query_encoder_model'],
        'query_encoder_prompt': c['encoder_query_prompt'],
        'query_encoder_source_fingerprint': query_fingerprint,
        'query_encoder_pooling': query_pooling})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--codec', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    prepare(a.config, a.codec, a.output, a.device)

if __name__ == '__main__': main()
