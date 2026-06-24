# This repository contains the code for [GCT-MARL](https://openreview.net/pdf?id=Trc7ZxmNAM) paper.

**Graph-Based Contrastive Transfer for Sample-Efficient Cooperative Multi-Agent Reinforcement Learning**

Animesh Animesh, Satheesh K Perepu, Kaushik Dey · *Accepted at the Continual RL Workshop, Reinforcement Learning Conference (RLC) 2026.*

---

## Repository layout

```text
src/
  obs_parser.py          # SMAC flat obs -> structured per-entity tensors
  entity_encoder.py      # variable obs -> fixed E-dim per-agent feature
  gcl_module.py          # MAIL 3-view graph-contrastive module (SGC + InfoNCE)
  agent.py               # entity_encoder + GRU + GCL + Q-head (TransferAgent)
  qmix.py                # QMIX mixing network
  buffer.py              # episode replay buffer
  runner.py              # per-step env rollout, visibility-based adjacency
  learner.py             # source learner:   L_TD + β·L_GCL
  transfer_learner.py    # transfer learner: L_TD + β·L_GCL + L_xfer (learnable α)
  main_source.py         # Phase 1 entry point
  main_transfer.py       # Phase 2 entry point (loads a source checkpoint)
  main_continual.py      # continual sequence entry point + backward eval
  evaluate.py            # load any checkpoint and report greedy win rate

scripts/
  run_source.sh          # one map -> one source run
  run_transfer.sh        # one (src, tgt) pair -> one transfer run
  launch_all_source.sh   # all source maps across multiple GPUs.
  launch_continual.sh    # one continual sequence

results/                 # bundled checkpoints + eval.csv for the paper scenarios
  source/<map>_seed<S>/
  transfer/<src>_to_<tgt>_seed<S>/
  continual/continual_marines_seed0/
  ablation*/             # per-view / learnable-alpha ablation curves.
```

## Installation

GCT-MARL needs PyTorch and the StarCraft II game + SMAC maps.

**1. Create an environment and install Python deps.**

```bash
conda create -n gct-marl python=3.10 -y
conda activate gct-marl
pip install -r requirements.txt
pip install "protobuf<3.21
```

**2. Install SMAC Environemnt (the oxwhirl StarCraft Multi-Agent Challenge).**

```bash
pip install "git+https://github.com/oxwhirl/smac.git"
```

**3. Install StarCraft II + the SMAC maps.**

- Install the StarCraft II game. On Linux, use Blizzard's headless package
  (see the [SMAC instructions](https://github.com/oxwhirl/smac#installing-starcraft-ii)).
- Download `SMAC_Maps` and place it under `$SC2PATH/Maps/`.
- Point SMAC at the install:

```bash
export SC2PATH=~/StarCraftII          # wherever StarCraft II is installed
````

## Quick start: evaluate the saved checkpoints

Load the bundled `3m → 8m` transfer agent and plays 32 greedy episodes:

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python src/evaluate.py --ckpt results/transfer/3m_to_8m_seed0 --episodes 32 --device cuda:0
```

`--ckpt` accepts either a run directory (its best checkpoint is selected from
`top_ckpts.json`) or a specific `.pt` file. CPU-only works with `--device cpu`.

## Phase 1 — Source training

Train the full GCT-MARL agent + QMIX mixer from scratch on a single SMAC map.

```bash
# python src/main_source.py --map <MAP> --run-name <NAME> --device cuda:0 --t-max <STEPS>
python src/main_source.py --map 3m --run-name 3m_seed0 --seed 0 --device cuda:0 --t-max 1000000
```
Or via the wrapper (`<map> <gpu> [seed] [t_max]`):

```bash
bash scripts/run_source.sh 3m 0 0 1000000          # one map
bash scripts/launch_all_source.sh                  # all source maps across multiple GPUs
```

## Evaluating a checkpoint

`src/evaluate.py` loads any checkpoint produced by this repo (source, transfer,
or continual), rebuilds the agent, and runs greedy (ε = 0) episodes:

```bash
# point at a run directory -> evaluates its best checkpoint
python src/evaluate.py --ckpt results/source/3m_seed0 --episodes 32

# or a specific checkpoint file
python src/evaluate.py --ckpt results/source/8m_seed0/ckpt_t380605_w1.000.pt --episodes 32

# a continual phase checkpoint: pass --map (its config stores the whole sequence)
python src/evaluate.py --ckpt results/continual/continual_marines_seed0/phase_3_8m_vs_9m --map 8m_vs_9m
```

## Phase 2 — Transfer training

```bash
python src/main_transfer.py \
    --source-ckpt results/source/3m_seed0/ckpt_t350613_w1.000.pt \
    --target-map 8m \
    --learn-alphas \
    --run-name 3m_to_8m_seed0 --seed 0 --device cuda:0 --t-max 1000000
```

Or via the wrapper (`<src_map> <tgt_map> <gpu> [seed] [t_max] [extra...]`), which
resolves the best source checkpoint automatically and injects `--learn-alphas`:

```bash
bash scripts/run_transfer.sh 3m 8m 0 0 1000000
```

## Continual learning

Train a sequence of maps with a shared backbone; at each phase `k > 1` the
previous phase's backbone is the frozen source for `L_xfer`. After each phase,
the model is evaluated on all prior maps (using the saved per-phase head plugged
into the current backbone) to measure backward transfer.

```bash
python src/main_continual.py \
    --maps "3m,8m,8m_vs_9m,10m_vs_11m" \
    --source-ckpt results/source/3m_seed0/ckpt_t350613_w1.000.pt \
    --t-max-per-phase 1000000 \
    --run-name continual_marines_seed0 --seed 0 --device cuda:0
```

Or via the wrapper (`<run-name> <maps_csv> [gpu] [seed] [tmax_per_phase] [extra...]`,
runs in a detached `screen` session):

```bash
bash scripts/launch_continual.sh continual_marines_seed0 \
     "3m,8m,8m_vs_9m,10m_vs_11m" 0 0 1000000 \
     --source-ckpt results/source/3m_seed0/ckpt_t350613_w1.000.pt
```

## Citation
----
If you find this repository useful, please cite:

```bibtex
@inproceedings{
animesh2026gctmarl,
title={{GCT}-{MARL}: Graph-Based Contrastive Transfer for Sample-Efficient Cooperative Multi-Agent Reinforcement Learning},
author={Animesh Animesh and Satheesh K Perepu and Kaushik Dey},
booktitle={Continual Reinforcement Learning Workshop at RLC 2026},
year={2026},
url={https://openreview.net/forum?id=Trc7ZxmNAM}
}
```

GCT-MARL builds directly on **MAIL** (Du et al., *Multi-Agent Communication with Information Preserving Graph Contrastive Learning*, IJCAI 2025) and the SMAC benchmark (Samvelyan et al., AAMAS 2019); please consider citing MAIL as well.