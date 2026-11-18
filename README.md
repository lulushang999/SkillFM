# SkillFM

Implementation of a classic skill Codec and improved MeanFlow (iMF).
The Python package is named `skillflow`; use that name in the commands below.
The skill encoder learns reconstruction; the original query encoder remains frozen.
iMF generates a skill latent with **NFE=1** (one interval-average flow evaluation
with `r=0` and `t=1`),
followed by one greedy skill decode and, for a valid generated skill, one agent rollout.

Only main-method code, configuration, and one training-skill version per domain are
included. No model weights, test data, trajectories, results, or experiment logs are bundled.

| Training file | Rows | Evaluation domains |
|---|---:|---|
| `data/skills/alfworld/` | 3553 | ALFWorld Seen / Unseen |
| `data/skills/nq.jsonl` | 4000 | NQ / TriviaQA / PopQA |
| `data/skills/multihop.jsonl` | 2723 | HotpotQA / 2WikiMultiHopQA / MuSiQue / Bamboogle |
| `data/skills/webshop.jsonl` | 2000 | WebShop |

The ALFWorld directory contains two JSONL shards of the same training library;
loaders read them in filename order. Other domains use one JSONL file each.

Each row contains a public `task_type` and `canonical_goal`, an `executable_skill`
JSON object with three guidance fields, and training identifiers.

## Setup

Use Python 3.10+ with a CUDA-compatible PyTorch installation:

```bash
pip install -e .
pip install -e '.[inference]'
```

Set model paths in `configs/<domain>.yaml`, `configs/actors/<domain>.yaml`, and
`configs/summarizer.yaml`. Supply the Qwen3 embedding model, instruction decoder,
and actor checkpoints separately. Install ALFWorld with `pip install -e '.[alfworld]'`
and prepare its official environment data for ALFWorld evaluation; set `ALFWORLD_DATA`
to the data root referenced by `configs/actors/alfworld_environment.yaml`.

## Train and generate

Run commands from this directory. Use a new output directory for each run.
This example uses the NQ recipe; substitute the other domain configuration as needed.

```bash
python -m skillflow.train_codec --config configs/nq.yaml --validate-only --world-size 4
torchrun --nproc_per_node=4 -m skillflow.train_codec --config configs/nq.yaml --output runs/nq/codec
python -m skillflow.prepare_latents --config configs/nq.yaml --codec runs/nq/codec --output runs/nq/pairs
torchrun --nproc_per_node=4 -m skillflow.train_flow --config configs/nq.yaml --codec runs/nq/codec --pairs runs/nq/pairs --output runs/nq/flow
python -m skillflow.generate --config configs/nq.yaml --codec runs/nq/codec --checkpoint runs/nq/flow/flow_best.pt --tasks tasks.jsonl --output runs/nq/assignments.jsonl
```

Recommended GPU counts are recorded separately for Codec and Flow in each config
(ALFWorld: 3 and 6; other domains: 4). Global batches are preserved when changing
world size, but batch tails and random streams may change. NQ trains Flow with a
single global batch of 144 for all 30,000 steps. ALFWorld, Multihop and NQ hold out
complete task conditions from Flow training and select checkpoints on that holdout;
WebShop uses a training-subset latent probe. Multihop scores candidates using decoded
rule features and latent geometry; the other domains use latent geometry. All selection
and final generation use NFE=1. Training consumes only the bundled training skills.

## Evaluate

Task JSONL needs `task_id` and a public `canonical_goal`, `goal`, `query`, or `question`.
ALFWorld also needs `task_type` and `game_file`; WebShop needs integer `session`;
QA needs `dataset` and `golden_answers` for scoring. WebShop goals must exactly
match the instruction returned by the selected environment session. Gold-answer fields are excluded from
skill-generation inputs. Keep official splits and task order fixed when comparing runs.

```bash
python -m skillflow.evaluation --domain nq --config configs/actors/nq.yaml --tasks tasks.jsonl --assignments runs/nq/assignments.jsonl --output runs/nq/evaluation.jsonl
```

Use `--domain alfworld`, `multihop`, or `webshop` with its matching actor config.
Actor output budgets per action remain domain-specific: ALFWorld 8192 tokens/50 actions, NQ
2048/4, Multihop 8192/4, and WebShop 4096/30.
Invalid generated skills remain failures in the complete denominator. Multihop
retains grounded guidance filtering and explicitly records any question-only fallback.

QA requires a separately prepared Wikipedia corpus, E5 index, and GPU-enabled FAISS:

```bash
pip install -e '.[retrieval]'
python -m skillflow.evaluation.services.retrieval --index-path data/wiki.index --corpus-path data/wiki.jsonl --model-path models/e5-base-v2 --gpu-map 0,1,2,3,0,1,2,3
```

For WebShop, install its official repository, catalog, and Lucene index in its own
compatible environment, then start the bridge using this package on `PYTHONPATH`:

```bash
python -m skillflow.evaluation.services.webshop --webshop-root external/WebShop --index-path external/WebShop/search_engine/indexes
```

The service addresses are configured in the actor YAML files. Benchmark assets
and their upstream licenses remain separate from this code release.

## Summarization model

Distill training trajectories into new skills with the included ALFWorld, QA,
and WebShop summarizers. Input rows contain `task_id`, public task fields,
`success`, and `steps` with `observation` and `action`; WebShop also needs `done`,
`score` in [0,1], and `session`. Set the model checkpoint in `configs/summarizer.yaml`.

```bash
python -m skillflow.summarization --dataset nq --input training-rollouts.jsonl --output new-training-skills.jsonl
```

`--validate-only` checks inputs without loading a model. Generated skills are
separate outputs; the supplied four training libraries are ready to load as-is.

## CPU checks

```bash
python -m unittest discover -s tests
```

The checks cover one-call NFE, iMF gradients, training data, and public-condition
isolation. Full GPU training and benchmark evaluation must be run with external
models and benchmark assets.
