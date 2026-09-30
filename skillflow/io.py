"""Input contracts shared by training and one-shot skill generation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import yaml


def load_config(path):
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    flow = cfg['flow']
    if type(flow.get('sampling_steps')) is not int or flow['sampling_steps'] != 1:
        raise ValueError('The release requires NFE=1')
    if flow.get('sampling_method') != 'meanflow_interval_average_velocity':
        raise ValueError('Expected the interval-average velocity sampler')
    if flow.get('meanflow_objective') != 'improved_boundary_velocity_vloss_v1':
        raise ValueError('Expected the improved MeanFlow boundary objective')
    expected = {
        'optimizer': 'adam', 'path_type': 'linear', 'cfg_mode': 'none',
        'time_sampler': 'logit_normal', 'prior_distribution': 'standard_normal',
        'data_scaling': 'unit_l2_times_sqrt_latent_dim',
        'condition_mode': 'task_type_goal_v1', 'gradient_clipping_mode': 'none',
    }
    for key, value in expected.items():
        if flow.get(key) != value:
            raise ValueError(f'Classic iMF requires flow.{key}={value!r}')
    if flow.get('grad_clip') not in (None, 0, 0.0):
        raise ValueError('Classic iMF does not apply gradient clipping')
    stages = flow.get('batch_schedule', [])
    next_step = 1
    for stage in stages:
        if stage['start_step'] != next_step or stage['end_step'] < stage['start_step']:
            raise ValueError('Flow batch stages must be contiguous, ordered, and nonoverlapping')
        next_step = stage['end_step'] + 1
    if stages and next_step != flow['training_steps'] + 1:
        raise ValueError('Flow batch stages must cover all training steps')
    return cfg


def jsonl_paths(path):
    """Resolve a JSONL file or a directory of shards in filename order."""
    source = Path(path)
    if not source.is_dir():
        return [source]
    parts = sorted(p for p in source.glob('*.jsonl') if p.is_file())
    if not parts:
        raise ValueError('Input directory has no JSONL shards')
    return parts


def read_jsonl(path):
    rows = []
    for part in jsonl_paths(path):
        with part.open(encoding='utf-8') as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    if not rows:
        raise ValueError('Input JSONL is empty')
    ids = [str(row['task_id']) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('task_id values must be unique')
    return rows


def training_rows(cfg):
    from skillflow.skills.executable import canonicalize_executable_skill
    rows = read_jsonl(cfg['data']['skills'])
    if len(rows) != int(cfg['data']['expected_count']):
        raise ValueError('Training skill count does not match config')
    for row in rows:
        dataset = {'hotpotqa': 'multihop', 'naturalquestions': 'nq',
                   'natural_questions': 'nq'}.get(row.get('dataset'), row.get('dataset'))
        if dataset != cfg['dataset']:
            raise ValueError('Training skill dataset does not match config')
        if row.get('source_split') != 'train':
            raise ValueError('Only training skills may enter training')
        if not row.get('task_type') or not row.get('canonical_goal'):
            raise ValueError('Missing public task condition')
        canonicalize_executable_skill(row['executable_skill'])
    return rows


def sha256(path):
    """Hash a file, or the exact concatenated bytes of ordered JSONL shards."""
    h = hashlib.sha256()
    for part in jsonl_paths(path):
        with part.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
