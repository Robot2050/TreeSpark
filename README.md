<h1 align="center">
    TreeSpark: Learning to Draft the Trees You Verify for Speculative Decoding
</h1>

<p align="center">
  <img src="docs/framework.jpg" alt="TreeSpark framework" width="95%">
</p>

## 🎯 Who Should Pay Attention to Our Work?

* **LLM Inference and Serving Engineers:** Parallel drafters such as DFlash and DSpark lose the remainder of each block to the first rejection. TreeSpark turns the same backbone computation into a budgeted tree of alternatives, so more of every computed block survives verification.
* **Speculative Decoding Researchers:** Reference-conditioned training never supervises the parent states or the candidate selections that tree verification actually uses. TreeSpark closes this gap by replaying the deployed candidate policy during training.
* **Draft Distillation Practitioners:** TreeSpark provides inference-aligned supervision without placing a frozen target model in the training loop. Both tree objectives reuse the reference-chain target distributions already computed for distillation.
* **High-Concurrency Serving Developers:** The bundled continuous-batching engine packs and verifies all in-flight draft trees in a single target forward per round, with device-side mask assembly and vectorized acceptance retrieval.

## ✨ Core Contributions

1. **Parent-Conditioned Tree Drafting from One Backbone Pass.**
   One draft backbone forward produces vocabulary logits for every draft depth, and a lightweight first-order Markov correction head specializes each depth to a specific parent token (`z_d(u) = ℓ_d(x) + W·e(t_u)`). Branching therefore requires only extra head evaluations and never repeats the backbone. Trees expand with branching factor `k=4` over `B=7` depths and are packed under a global budget of `N=32` non-root nodes: up to 100 candidates before packing, and 33 target input positions including the anchor.

2. **Training under the Deployment Policy.**
   Training replays the exact candidate policy used at serving time, namely per-parent top-k expansion, joint path-score frontier pruning, and global top-N packing, on model-generated parent states. Discrete selections (token identities, frontier choices, packing decisions) are detached, while the conditional distributions remain differentiable through the backbone and the Markov head. This removes the exposure mismatch between teacher-forced training and tree deployment without any online target rollouts.

3. **Complementary Survival and Coverage Objectives.**
   Survival NLL weights the likelihood of each packed edge by its reference-derived path mass `m(v)`, a differentiable surrogate for how deeply target probability mass survives inside the verified tree. The capped coverage hinge (`J=4`, margin `δ=1`, cap `c=4`, depth decay `γ=4`) promotes target-important tokens near the local top-k admission boundary, including candidates that would otherwise be omitted and receive no survival signal. The two terms act at complementary stages of selection: coverage supervises expanded parents before the node budget is applied, and survival supervises the edges retained after packing.

4. **Exactness by Construction, Zero Extra Cost.**
   Greedy tree verification preserves autoregressive greedy output because every committed token is decided by the target model (prefix-closed packing, tree attention, and an argmax-following acceptance walk). The tree objectives add no target evaluations during training and no model components at inference.

## 📊 Results

We evaluate greedy decoding with **Qwen3-4B** and **Qwen3-8B** targets on nine general benchmarks (GSM8K, MATH-500, AIME 2025, HumanEval, MBPP, LiveCodeBench, MT-Bench, Alpaca, ArenaHard-v2) and six writing domains from WritingBench.

* **Acceptance length.** TreeSpark improves mean acceptance length over DSpark by **23.4% to 31.6%** across targets and suites (4.23 to 5.25 on the Qwen3-4B general suite, +24.1%; writing domains +31.4% and +31.6%).
* **Single-request throughput.** Gains of **15.1% to 25.0%** over DSpark. The longer accepted paths more than offset the cost of constructing and verifying a branching proposal.
* **Batched serving.** On Qwen3-4B, mean system throughput improves by **12.6% to 17.2%** at total concurrency C ∈ {6, 12, 36}, and per-request decoding throughput improves by 11.8% to 17.6%.
* **Isolated supervision effect.** Against the capacity- and policy-matched DSpark + Tree control on Qwen3-4B, tree supervision alone raises mean acceptance length from 4.93 to 5.25 (**+6.4%**) and mean system throughput by **4.2%** at C=12 and **5.0%** at C=36.
* **Ablations at C=12.** Removing survival supervision reduces the throughput gain from 17.2% to 11.9%, and removing coverage reduces it to 16.0%. Widening the draft attention head dimension from 128 to 192 adds a further 2.0% system throughput at fixed objectives and candidate policy.

## 🛠️ Quick Start

### Installation

```bash
python -m pip install -r requirements.txt
```

The code runs on Ascend NPUs; the reported experiments use Ascend 910B.

### Data Preparation

Training consumes a target cache: reference continuations with cached target hidden states and aligned target logits for each prompt, so the frozen target model is never needed during training. The loader is `deepspec/data/target_cache_dataset.py`, and the cache is produced with the target-cache preparation pipeline inherited from DeepSpec. The training mixture comprises 50,000 prompts spanning mathematics (29.9%), code (40.2%), dialogue and instructions (19.9%), and writing (10.0%), at a maximum sequence length of 4,096 tokens.

### Training

```bash
bash scripts/train/train.sh            # single node
bash scripts/train/train_multinode.sh  # two nodes (8+8)
```

Key settings (see `config/dspark/dspark_qwen3_4b.py`), matching the reference configuration used in the paper:

| Setting | Value |
|---|---|
| Draft architecture | 5-layer semi-autoregressive drafter, attention head dim 192 |
| Markov correction | first-order, rank r = 256 |
| Block size B / branching k / budget N | 7 / 4 / 32 |
| Tree anchors per sequence | 128 (of 512 backbone anchor blocks) |
| Loss weights | CE 0.1, L1 0.9, confidence 1.0, survival 0.1, coverage 0.1 |
| Optimizer | AdamW, lr 6e-4, 4% warmup, bf16, global batch 64, 10 epochs |

The replay hyperparameters (`tree_loss_top_k`, `tree_loss_max_nodes`) are recorded in the checkpoint config, and deployment must use matching `--tree-top-k` and `--tree-max-nodes` values. Setting `tree_loss_num_anchors=0` disables tree supervision entirely and yields the reference-trained DSpark + Tree control when combined with tree inference. `tree_survival_alpha` and `tree_rank_alpha` toggle the two objectives independently.

### Evaluation

All launchers live under `scripts/eval/`. Set `TARGET_MODEL` and `DRAFT_MODEL` through environment variables or at the top of each script.

```bash
# Sequential single-request evaluation
bash scripts/eval/eval.sh        # chain mode
bash scripts/eval/eval_tree.sh   # tree mode (k=4, N=32, the full TreeSpark policy)

# High-concurrency batched evaluation (closed-loop continuous batching;
# MODE=tree|chain|baseline, CONCURRENCY="1,6,12", JSON metrics dump)
bash scripts/eval/eval_high_concurrency.sh            # Ascend NPU
bash scripts/eval/eval_high_concurrency_multicard.sh  # multi-card (one closed loop per card)
```

Tree mode requires a greedy target (`--temperature 0`) and disables confidence-based truncation. Evaluation generates up to 2,048 tokens per request with no input truncation. Acceptance length counts committed tokens per verification round, including the target bonus token when emitted.

Benchmark prompts ship in `eval_datasets/`: the nine general benchmarks and the six WritingBench domains. Uncomment the writing entries in `eval.py::TASKS` to include them.

## 📁 Repository Structure

```
├── train.py                       # Training entry (spawns one worker per visible accelerator)
├── eval.py                        # Evaluation entry: sequential and batched engines
├── config/dspark/                 # Experiment configs (Qwen3-4B / Qwen3-8B targets)
├── deepspec/
│   ├── data/                      # Target-cache dataset, JSONL parsing, async prefetch
│   ├── modeling/dspark/           # Draft model (Qwen3), Markov heads,
│   │                              #   baseline + tree objectives (loss.py), tree replay (tree_loss.py)
│   ├── modeling/eagle3/           # EAGLE-3 baseline modeling
│   ├── trainer/                   # Training loop, checkpointing
│   └── eval/
│       ├── dspark/                # Sequential evaluator, chain/tree drafting ops (tree_ops.py)
│       ├── eagle3/                # EAGLE-3 sequential evaluator
│       └── batched/               # High-concurrency engine: continuous batching, ragged
│                                  #   batch KV, batched chain/tree proposal & verification
├── scripts/train/                 # Single-node and multi-node training launchers
├── scripts/eval/                  # Sequential / tree / high-concurrency eval launchers
└── eval_datasets/                 # Benchmark prompts (general suite + WritingBench domains)
```

## 📄 License

TreeSpark is released under the [MIT License](./LICENSE). It includes code adapted from third-party projects under their own licenses; see [NOTICE](./NOTICE) for the full attribution.
