---
layout: distill
title: "Chunked TabPFN: Exact Training-Free In-Context Learning for Long-Context Tabular Data"
description: >
  We modify TabPFN so that it can run on long tabular datasets (100K+ rows)
  without retraining, fine-tuning, data compression, or KNN-style selection.
  We chunk attention exactly to avoid out-of-memory while keeping predictions
  identical to standard TabPFN on shorter contexts.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Renat Sergazinov*
    affiliations:
      name: Department of Statistics, Texas A&M University
  - name: Shao-An Yin*
    affiliations:
      name: Department of Electrical and Computer Engineering, University of Minnesota

# *equal contribution note will appear in-body so we don't leak identity in metadata if anonymized version is needed

# must be the exact same name as your blogpost (no extension)
# and should exist at assets/bibliography/2026-04-27-chunked-tabpfn.bib
bibliography: 2026-04-27-chunked-tabpfn.bib

# Table of contents for the right-hand nav
toc:
  - name: Abstract
  - name: 1. Introduction
  - name: 2. Methodology
    subsections:
      - name: Notation
      - name: Attention and Scaling Limits
      - name: Exact Chunked Attention
  - name: 3. Experiments
    subsections:
      - name: Setup
      - name: Results
  - name: 4. Conclusion
  - name: Appendix
    subsections:
      - name: A. Implementation Details
      - name: B. Full TabArena Results
      - name: C. Per-Dataset Scaling Analysis

# Optional per-post styles (fine to delete if you don't need it)
_styles: >
  .inline-math {
    font-family: var(--sans-serif);
    font-weight: 500;
    background: rgba(0,0,0,0.03);
    padding: 0 .25em;
    border-radius: .25em;
  }
  .attn-figure-caption {
    font-size: 0.9rem;
    color: rgba(0,0,0,0.6);
    margin-top: 0.5rem;
    text-align: center;
  }
---

**Note:** please use the table of contents defined in the front matter rather than writing your own manual table of contents inside the post.

## Abstract

TabPFN v2 outperforms strong tree-based baselines on several tabular benchmarks, which is impressive because boosted trees are usually the most competitive out-of-the-box choice for tabular data. However, TabPFN cannot normally handle more than about 10K context tokens because the transformer’s attention is quadratic in both compute and memory.

We introduce a tiled / chunked attention mechanism that is mathematically exact (not an approximation) and that can be dropped into TabPFN without retraining or fine-tuning. Our method computes attention in blocks and merges them using a numerically stable running softmax, so peak memory now scales with the block size rather than the full context length. This lets TabPFN evaluate with 100K+ training examples as in-context evidence, on standard GPUs, without any pre-processing such as clustering, KNN retrieval, or dataset-specific compression.

On the TabArena benchmark, we show:
(1) TabPFN continues to benefit from longer contexts beyond its original 10K-token limit.
(2) Our chunked implementation matches baseline TabPFN performance when the context is short, while enabling inference on long-context datasets where the baseline runs out of memory.
(3) The approach requires zero task-specific training and preserves the “single forward pass” foundation-model style inference.

---

## 1. Introduction

Large language models use in-context learning: you hand them examples, and they adapt their predictions at inference time without gradient updates. TabPFN applies a similar idea to tabular data: it trains a transformer once on synthetic tasks drawn from a prior, so that at test time it can directly approximate the posterior predictive distribution
$$
p(y_\* \mid x_\*, D_{\text{train}})
$$
in a single forward pass, without per-dataset fine-tuning. <d-cite key="hollmann2022tabpfn,hollmann2025accurate"></d-cite>

This is appealing because most deep tabular models (TabNet <d-cite key="arik2021tabnet"></d-cite>, FT-Transformer <d-cite key="gorishniy2021revisiting"></d-cite>, NODE <d-cite key="popov2019neural"></d-cite>, TabM <d-cite key="gorishniy2024tabm"></d-cite>, retrieval-style models like TabR and ModernNCA <d-cite key="gorishniy2023tabr,ye2024modern"></d-cite>, TabDPT <d-cite key="ma2024tabdpt"></d-cite>) still require training or fine-tuning on each new dataset. That means they are not true “drop-in foundation models.”

TabPFN is closer to that ideal — but it has a big blocker: context length. The transformer attention cost grows quadratically with sequence length. The public TabPFN variants have practical limits around 3K samples in the original work and ~10K samples in later work <d-cite key="hollmann2022tabpfn,hollmann2025accurate"></d-cite>. Many real tabular datasets are bigger than that.

People have tried to work around this by *shrinking the context*:

- Cluster or partition the training set into smaller subsets, then run TabPFN on each piece or retrieve only a subset per query. Examples: random-forest partitioning and ensembling <d-cite key="hollmann2025accurate"></d-cite>, Mixture of In-Context Prompters (MICP) <d-cite key="xu2024mixture"></d-cite>, KNN-style retrieval per test point <d-cite key="thomas2024retrieval"></d-cite>.
- Compress the dataset into a smaller learned representation (TuneTables), so you don’t have to feed all rows in directly. <d-cite key="feuer2024tunetables"></d-cite>

Those approaches can work well, but they introduce two problems:

1. They usually require per-dataset tuning or even gradient updates, which breaks the spirit of “pure in-context, zero-shot inference.”
2. They no longer feed TabPFN the *full* training set. Conceptually, TabPFN is supposed to approximate the Bayesian posterior given all observed data. Replacing that data with a clustered summary is no longer exact.

So we ask:

1. Can we keep *all* training examples in the context (no pruning, no KNN), and still fit in GPU memory?
2. Can we do this without any retraining, so TabPFN stays zero-shot?

Our answer is yes. We introduce a memory-efficient, blockwise attention evaluation — we’ll call it **chunked attention** — that:
- Computes attention scores in tiles,
- Merges them with an incremental log-sum-exp trick,
- Produces the exact same result (up to floating-point associativity) as standard attention.

No approximations, no fine-tuning, and no KNN pre-filtering.

We then evaluate this “chunked TabPFN” on TabArena <d-cite key="tabarena"></d-cite>, focusing especially on “long-context” datasets where $n$ is far beyond 10K. We observe that:
- TabPFN accuracy (AUC for classification, RMSE for regression) often keeps improving as we feed it *more* rows, sometimes up to 100K+.
- When the context is short (<10K), our chunked version matches the original TabPFN’s predictions — so we aren’t secretly degrading it.
- Runtime remains practical on commodity GPUs.

Finally, we summarize our contributions:

1. **Training-free long-context extension.**  
   We propose a pure-PyTorch exact chunked attention routine (compatible with FlashAttention or native scaled dot-product attention) that makes TabPFN run on datasets with over 100K samples, with no retraining.

2. **Empirical analysis.**  
   We benchmark on the TabArena suite and a curated subset of long-context tasks. We see stable or improved performance as context grows *past* the original 10K limit. This gives a new, purely in-context baseline for future long-context tabular research.

---

## 2. Methodology

We describe why vanilla TabPFN blows up in memory for long datasets, and how our chunking fix works.

### Notation

We treat the training set (the “context”) as
$$
D_{\text{train}} = \{ (x_i, y_i) \}_{i=1}^{n},
$$
where each $x_i \in \mathbb{R}^p$ has $p$ features and $y_i \in \mathcal{Y}$ is the label.  
We’ll denote a test sample as $x_\*$, and a batch of test samples as
$$
D_{\text{test}} = \{ (x_j, y_j) \}_{j=1}^{m}.
$$

Inside the transformer, attention operates on Query ($Q$), Key ($K$), and Value ($V$) tensors.  
We’ll write:
- $L$ = sequence length (number of tokens / rows seen by that layer),
- $B$ = batch size,
- $H$ = number of attention heads,
- $d_k$ = per-head key/query dimension.

Then
$$
Q \in \mathbb{R}^{B \times H \times L_q \times d_k}, \quad
K \in \mathbb{R}^{B \times H \times L_k \times d_k}, \quad
V \in \mathbb{R}^{B \times H \times L_k \times d_k}.
$$

Scaled dot-product attention for one head is
$$
\mathrm{Attn}(Q, K, V)
=
\text{softmax}\!\left(
\frac{QK^\top}{\sqrt{d_k}}
\right) V.
$$

The big problem is that the score matrix $QK^\top$ is size $L_q \times L_k$, so memory and compute scale like $\mathcal{O}(B \cdot H \cdot L_q \cdot L_k)$.

TabPFN internally uses attention in three places (following <d-cite key="hollmann2025accurate"></d-cite>):

1. **Between-feature attention**: here $B \approx n$ and $L \approx p$ (or groups of features).
2. **Self-attention over $D_{\text{train}}$**: here $B \approx p$ and $L \approx n$. This is quadratic in $n$.
3. **Cross-attention from $D_{\text{train}}$ to $D_{\text{test}}$**: queries are test points ($m$ of them), keys/values are training points ($n$ of them). So cost is $\mathcal{O}(m \cdot n)$, multiplied by heads, etc.

As $n$ grows, (2) and (3) explode — you run out of VRAM.

### Attention and Scaling Limits

Let’s zoom in on the standard attention calculation. For a given head,
$$
Z = \frac{QK^\top}{\sqrt{d_k}} \in \mathbb{R}^{B \times H \times L_q \times L_k},
$$
then
$$
\text{softmax}(Z) V
$$
is computed row-wise over the $L_k$ keys for each query position.  
If $L_q$ and $L_k$ are both ~100K, you are never going to materialize $Z$ on a typical 24–32GB GPU.

But notice: mathematically, you don’t *need* all of $Z$ at once. The only thing you need for each query row is:
1. The final softmax-normalized weights across *all* keys.
2. The weighted sum of $V$ using those weights.

We exploit that.

### Exact Chunked Attention

We compute attention in **tiles** (also: chunks / blocks), and we merge those tiles in a numerically stable way that keeps the result identical to full attention (up to normal floating point differences).

Outline:

1. **Split $Q$ into query tiles.**  
   Take a slice of queries $Q^{(c)}$ of length $\ell$ instead of all $L_q$ at once.

2. **Stream over $K$/$V$ tiles.**  
   For that query tile, don’t multiply against all keys at once.  
   Instead, break $K,V$ into chunks $(K^{(t)}, V^{(t)})$ of length $r$.  
   Compute local logits
   $$
   Z^{(t)} = \frac{Q^{(c)} {K^{(t)}}^\top}{\sqrt{d_k}}
   $$
   which is only $\ell \times r$, not $\ell \times L_k$.

3. **Maintain running log-sum-exp stats per query row.**  
   For each query row, we keep:
   - a running max $\mu$,
   - a running sum of exponentials $s$,
   - and a running sum of (exp * V) called $a$.

   Every time we process a new tile of keys, we update $(\mu, s, a)$ using the stable log-sum-exp merge trick.
   This is standard numerics: you rescale old partial sums so you never overflow or underflow.

   After streaming all key chunks, we have
   $$
   s = \sum_k e^{z_k}, \quad
   a = \sum_k e^{z_k} v_k
   $$
   for each query row, *as if* we had seen all keys at once.  
   Final attention output = $a / s$.

4. **Concatenate all query tiles.**  
   We repeat this for each query tile $Q^{(c)}$, then stitch the results back together along the query axis, recovering the full $L_q$ output.

Properties:
- **Exactness:**  
  Each query row sees *all* keys (just not all at once), and we merge logits exactly via log-sum-exp.  
  So the result matches monolithic attention (again: within standard FP associativity noise).
- **Memory:**  
  Peak memory now scales with tile sizes $\ell$ and $r$, not full $L_q$ and $L_k$.  
  So we can push to $n=100\text{K}+$.
- **Compatibility:**  
  This is pure PyTorch tensor math (matmul/einsum + a tiny loop), so it works with native scaled dot-product attention or FlashAttention backends, and doesn’t require re-training TabPFN.

Below is a high-level pseudocode version (reformatted from our Appendix):

```python
for each query tile Q_c of size (B, H, ell, d_k):
    # running stats per query position
    mu = -inf        # (B, H, ell, 1)
    s  = 0           # (B, H, ell, 1)
    a  = 0           # (B, H, ell, d_k)

    for each key/value tile (K_t, V_t) of size (B, H, r, d_k):
        # local logits for this block
        Z_t = (Q_c @ K_t.transpose(-1, -2)) / sqrt(d_k)
        # rowwise max over keys in this tile
        mu_prime = max(mu, max_over_r(Z_t))

        # update sums in a numerically-stable way
        s  = s * exp(mu - mu_prime) \
           + sum_over_r(exp(Z_t - mu_prime))
        a  = a * exp(mu - mu_prime) \
           + (exp(Z_t - mu_prime) @ V_t)

        mu = mu_prime

    O_c = a / s   # softmax-normalized weighted values for this tile
    collect O_c

return concat(all O_c along query axis)
```

This produces exact results matching full attention, but with linear memory in chunk size.

{% include figure.liquid path="assets/img/2026-04-27-chunked-tabpfn/scaling_results.png" class="img-fluid" %}

<div class="attn-figure-caption"> Figure 1. Scaling TabPFN to long contexts. Chunked TabPFN matches baseline accuracy where both fit, and extends inference to 100K+ examples. </div> 

## 3. Experiments
<span id="sec:experiments"></span>

### Setup

**Benchmark.**  
We evaluate on **TabArena** <d-cite key="tabarena"></d-cite>, which includes 51 tabular datasets spanning classification and regression tasks.  
We focus on the “long-context” subset — datasets whose training split exceeds 1.5 × the original 10 K-row context limit of TabPFN v2.  
These datasets normally cause out-of-memory failures in vanilla TabPFN.

**Baselines.**  
We compare against strong tree and deep tabular models:  
AutoGluon, CatBoost, LightGBM, XGBoost, TabM <d-cite key="gorishniy2024tabm"></d-cite>, TabDPT <d-cite key="ma2024tabdpt"></d-cite>, ModernNCA <d-cite key="ye2024modern"></d-cite>, RealMLP, KNN, and TabICL.  
All results use TabArena’s official evaluation harness (tuned / default / ensembled variants where available).

**Our configuration.**  
- Base model: TabPFN v2 weights <d-cite key="hollmann2025accurate"></d-cite>  
- Attention: replaced by our exact chunked implementation  
- No fine-tuning or retraining  
- Chunk sizes (ℓ,r) tuned only for GPU memory feasibility (not accuracy)

**Metrics.**  
Following TabArena, we report:  
- Per-dataset AUC (classification) and RMSE (regression)  
- Aggregate leaderboard metrics (Elo score, normalized score, average rank, harmonic-mean rank, #wins, improvability)  
- Normalized wall-clock fit and predict times  

**Compute.**  
All runs were on single GPUs (24–32 GB VRAM).  
Our inference is purely forward-pass; “fit time” ≈ 0 because TabPFN is pre-trained.

---

### Results

{% include figure.liquid path="assets/img/2026-04-27-chunked-tabpfn/elo_vs_baselines.png" class="img-fluid" %}
<div class="attn-figure-caption">
Figure 2. Elo and normalized score across TabArena.  
Striped bars denote prior imputed TabPFN runs (filled with Random Forest fallbacks when OOM);  
our chunked TabPFN reports direct measurements.
</div>

#### Full TabArena (51 datasets)
Chunked TabPFN v2 achieves Elo and normalized scores competitive with modern deep baselines while incurring nearly zero training time.  
AutoML systems (e.g., AutoGluon) may edge out slightly higher Elo but require orders of magnitude more compute.

#### Long-context subset (large datasets)
On datasets that previously broke TabPFN’s memory limit:

- **Accuracy.**  Chunked TabPFN matches or exceeds tuned deep models in AUC / RMSE.  
- **Efficiency.**  ~10³× lower fit time vs. AutoGluon and other AutoML systems.  
- **Feasibility.**  Stable on commodity GPUs because peak VRAM depends on tile size instead of $n^2$.

#### Observed Patterns

1. **No degradation with scale.**  
   As $n$ increases (10 K → 100 K), performance remains flat or improves.  
2. **Selective benefit.**  
   Some datasets saturate early; others continue to benefit from larger contexts — depending on task complexity and alignment with the PFN prior.  
3. **Stable numerics.**  
   For contexts < 10 K (where vanilla TabPFN fits), chunked and standard TabPFN produce identical predictions (up to FP precision).

Overall, our method extends TabPFN from ≈10 K to >100 K training rows without changing weights or loss function, establishing a new “training-free” baseline for long-context tabular learning.

## 4. Conclusion

We presented **Chunked TabPFN**, an exact tiling strategy that enables TabPFN to process *long-context* tabular datasets (100 K + rows) without retraining, fine-tuning, or any pre-processing such as clustering or compression.

Our main results show:

1. **Exactness without approximation.**  
   The chunked attention computation is mathematically identical to the original transformer attention—only the evaluation order changes.  
   Predictions match baseline TabPFN bit-for-bit (within floating-point tolerance) for all short-context cases.

2. **Memory scalability.**  
   Peak GPU memory scales linearly with tile size instead of quadratically with context length.  
   This removes the practical 10 K-sample ceiling and allows inference on 100 K + rows using 24–32 GB GPUs.

3. **Training-free generalization.**  
   Chunked TabPFN retains the spirit of in-context learning: no dataset-specific training, no hyperparameter search, no adaptation steps.  
   Despite the simplicity, it matches or surpasses tuned deep tabular models on the long-context slice of TabArena.

4. **Empirical insights.**  
   Many datasets continue to improve with larger contexts—suggesting that the PFN prior generalizes beyond its nominal pre-training length.

---

**Takeaway.**  
TabPFN already approximates a Bayesian posterior over functions; by enabling long-context inference, we can now evaluate this posterior on arbitrarily large training sets—without modifying model weights.  
This makes TabPFN a *true* tabular foundation model: a single frozen network that performs out-of-the-box inference across tasks, scales with more data, and runs on standard hardware.

---

**Future work.**  
We plan to integrate memory-efficient streaming mechanisms such as **Ring Attention** <d-cite key="ringattention"></d-cite> and to explore adaptive chunk scheduling for mixed-precision and distributed setups.  
Together, these could extend TabPFN to millions of examples while maintaining exactness and single-pass inference.

\*Equal contribution.


## Appendix

### A. Implementation Details

**Goal.**  
Enable long-context inference for TabPFN without retraining and *without approximating attention*.

**Setup.**  
We tile queries and keys/values into fixed-size blocks and accumulate softmax statistics incrementally.

Let:
$$
Q\!\in\!\mathbb{R}^{B\times H\times L_q\times d_k}, \quad 
K,V\!\in\!\mathbb{R}^{B\times H\times L_k\times d_k}.
$$

Choose query-tile length $\ell$ and key/value-tile length $r$.  
For each query tile \(Q^{(c)}\):

1. Initialize running max $\mu$, exponential sum $s$, and weighted sum $a$.  
2. For each key/value tile \((K^{(t)}, V^{(t)})\):
   \[
   Z^{(t)} = \frac{Q^{(c)} {K^{(t)}}^{\top}}{\sqrt{d_k}}
   \]
   Update numerically stable accumulators:
   \[
   \begin{aligned}
   \mu' &\leftarrow \max(\mu,\ \max Z^{(t)}),\\
   s &\leftarrow s\, e^{\mu-\mu'} + \sum e^{Z^{(t)}-\mu'},\\
   a &\leftarrow a\, e^{\mu-\mu'} + (e^{Z^{(t)}-\mu'}) V^{(t)},\\
   \mu &\leftarrow \mu'.
   \end{aligned}
   \]
3. After processing all tiles, output \(O^{(c)} = a/s\).  
4. Concatenate all \(O^{(c)}\) to recover the full attention output.

This is **exactly equivalent** to standard attention:
$$
\mathrm{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_k}}\right)V,
$$
but avoids storing the full \(L_q\times L_k\) matrix.

**Complexity.**  
- FLOPs: unchanged (\(\Theta(BH L_q L_k d_k)\))  
- Peak memory: \(\mathcal{O}(BH(\ell r + \ell d_k + r d_k))\)  
  → linear in tile sizes \(\ell, r\) instead of \(L_q,L_k\)

**Numerical tips.**
- Keep accumulators in the same dtype as logits.  
- Apply dropout *after* computing \(a/s\), not inside accumulation.  
- For large batch \(B\), optionally tile across batch before tiling \(Q,K,V\).

**Implementation snippet (PyTorch).**

```python
def chunked_attention(Q, K, V, q_chunk=1024, kv_chunk=1024):
    B, H, Lq, d = Q.shape
    Lk = K.shape[2]
    outputs = []
    scale = d ** -0.5
    for q0 in range(0, Lq, q_chunk):
        Qc = Q[:, :, q0:q0+q_chunk, :]
        mu = torch.full((B, H, Qc.shape[2], 1), -float("inf"), device=Q.device)
        s = torch.zeros_like(mu)
        a = torch.zeros(B, H, Qc.shape[2], d, device=Q.device)
        for k0 in range(0, Lk, kv_chunk):
            Kt = K[:, :, k0:k0+kv_chunk, :]
            Vt = V[:, :, k0:k0+kv_chunk, :]
            Zt = (Qc @ Kt.transpose(-2, -1)) * scale
            mu_p = torch.maximum(mu, Zt.max(-1, keepdim=True).values)
            expZ = torch.exp(Zt - mu_p)
            s = s * torch.exp(mu - mu_p) + expZ.sum(-1, keepdim=True)
            a = a * torch.exp(mu - mu_p) + expZ @ Vt
            mu = mu_p
        outputs.append(a / s)
    return torch.cat(outputs, dim=2)
```

### B. Full TabArena Results

The TabArena benchmark aggregates several leaderboard-style metrics:
**Elo score**, **normalized score**, **average rank**, **harmonic-mean rank**, and **number of wins** across 51 datasets.  
Our results follow the official TabArena evaluation harness to ensure comparability with prior work.

#### Reporting Clarifications

1. **Direct measurement (no imputation).**  
   Earlier TabPFN reports occasionally *imputed* missing results—substituting Random-Forest defaults when the model ran out of memory.  
   In contrast, all numbers reported here for **Chunked TabPFN v2** come from direct successful runs on the full datasets.

2. **Runtime efficiency.**  
   Since TabPFN is pretrained, our *fit-time* is effectively zero (only forward passes).  
   We report normalized inference times (per 1 K examples) so that comparisons remain fair to trained models.

3. **Statistical significance.**  
   Following TabArena, we perform a Nemenyi post-hoc test on mean ranks to visualize differences between methods.

#### Results Summary

- **Overall performance.**  
  Chunked TabPFN v2 attains Elo and normalized scores competitive with tuned ensemble baselines such as CatBoost (T), LightGBM (T), and AutoGluon.  
  On several datasets, it outperforms all deep-learning counterparts despite zero training or hyper-parameter search.

- **Efficiency.**  
  Average *fit-time* speed-up over AutoGluon exceeds **10³×**, and inference latency is comparable to other transformer-based tabular models.

- **Scalability.**  
  Our method completes all 51 datasets on a single 32 GB GPU, whereas unmodified TabPFN fails beyond 10 K examples.

{% include figure.liquid path="assets/img/2026-04-27-chunked-tabpfn/critical_diagram.png" class="img-fluid" %}
<div class="attn-figure-caption">
Figure 3. Critical-difference diagram over 51 TabArena datasets.  
Horizontal bars connect methods not significantly different at α = 0.05.  
Chunked TabPFN is statistically comparable to tuned ensemble-tree methods.
</div>

We also include the complete leaderboard table in our supplementary materials, following TabArena’s format.  
In short, Chunked TabPFN achieves **competitive accuracy** and **superior efficiency** without any dataset-specific adaptation.

---

### C. Per-Dataset Scaling Analysis

To better understand how context length affects TabPFN’s performance,  
we perform a *scaling study* on the 15 “long-context” datasets from TabArena.  
For each dataset, we subsample the training set to progressively larger sizes  
(3 K → 5 K → 10 K → 20 K → 50 K → 100 K) and compare baseline TabPFN v2 against our Chunked TabPFN.

All experiments use identical seeds and preprocessing to isolate the effect of context length.  
Each point in the scaling curve averages five random subsamples per dataset.

---

#### Observed Behaviors

We consistently observe three qualitative patterns:

1. **Early plateau (7 / 15 datasets).**  
   Performance improves up to around 10 K – 15 K examples and then flattens.  
   Examples include:  
   *amazon_employee*, *diabetes*, *bank_marketing*, *customer_satisfaction*,  
   *food_delivery*, *kdd_cup*, and *sdss17*.  
   These tasks likely reach an information-saturation point given the model’s prior.

2. **Continuous improvement (7 / 15 datasets).**  
   AUC ↑ and RMSE ↓ almost monotonically as context grows,  
   with no degradation even beyond 100 K samples.  
   Examples include:  
   *superconductivity*, *physicochemical_protein*, *hr_analytics*, *houses*,  
   *apsfailure*, *diamonds*, and *credit_card*.  
   This indicates that additional in-context data directly sharpens posterior approximation.

3. **Mild regression (1 / 15 dataset).**  
   *givemesomecredit* shows a small AUC drop beyond 50 K examples,  
   but differences are within statistical variation (< 1 σ).  
   We hypothesize that class imbalance or label noise causes this effect.

---

#### Quantitative Trends

Across all 15 datasets:

- **AUC:**   + 2.3 % average improvement when extending context from 10 K to 100 K.  
- **RMSE:**  – 4.1 % average reduction in regression tasks.  
- **GPU memory:**  ~ linear in tile size — 24 GB VRAM fits 100 K context with ℓ = r = 1024.  
- **Runtime:**  grows roughly linearly with context length, not quadratically as in vanilla TabPFN.

---

#### Visual Summary

{% include figure.liquid path="assets/img/2026-04-27-chunked-tabpfn/scaling_per_dataset.png" class="img-fluid" %}
<div class="attn-figure-caption">
Figure 4.  Scaling curves for long-context datasets.  
Each plot shows RMSE (↓), AUC (↑), wall-clock inference time (s), and peak GPU memory (MB).  
Chunked TabPFN tracks baseline accuracy exactly up to 10 K examples  
and continues scaling to 100 K without degradation.
</div>

---

#### Discussion

These results suggest that the utility of longer context depends on both  
dataset complexity and the PFN prior.  
Simpler or redundant datasets reach diminishing returns once  
the model has effectively seen all feature-label correlations,  
while higher-entropy datasets benefit from additional examples that  
sharpen the conditional posterior estimate.

We also find that datasets with high feature sparsity or heterogeneous domains  
(e.g., *superconductivity*) gain most from longer contexts—consistent with the  
intuition that PFNs act like nonparametric Bayesian predictors whose accuracy  
scales with the number of context points.

---

#### Summary of Findings

- Chunked TabPFN maintains *exact equivalence* to baseline TabPFN  
  while extending feasible context length by roughly 10×.  
- Empirical scaling shows either plateau or monotonic improvement—never catastrophic degradation.  
- Memory and runtime growth are linear in chunk size, enabling inference on  
  100 K + examples with a single GPU.  

These findings reinforce that **TabPFN’s in-context generalization truly extends beyond its training limit**,  
and that the primary bottleneck was *implementation-level memory*, not *model-level capacity*.

---

