# RESEARCH BRIEF: Learned per-block learning rates for SGD via anchored / horizon hypergradients

> **Hand-off document for a Claude Code research agent.**
> Project owner: a PhD student in ML/CV whose research covers optimization, regularization and representation learning, and specifically methods that steer DNN training toward flatter minima and better generalization (embodied AI is a later application area).
> Written 2026-10-31. Target machine: one Ubuntu workstation with **2× NVIDIA RTX 5080** (16 GB each, Blackwell consumer, compute capability sm_120, PCIe only, no NVLink). Both GPUs are fully available.
> Goal: run serious experiments that test, validate, refactor and tighten this research idea until it yields publishable results.

---

## 0. How to use this file

* Put this file at the repo root as `RESEARCH_BRIEF.md`. Create `CLAUDE.md` with the content in §0.2: it is short, it is loaded every session, and it points back here.
* Read the whole brief once before writing code. §3 has the math, §5 the known weaknesses, §6 the hypotheses, §8 the experimental program, and Appendix A the reference code (already verified).
* **Status labels used below**
  * **[Verified]** means derived and checked numerically (float64 finite differences, exact trajectory equivalence, or Monte Carlo).
  * **[Prediction]** means derived from theory but never tested at realistic scale. Treat these as hypotheses to confirm or kill.
  * **No empirical result at realistic scale exists yet.** Earlier CPU toy runs were deliberately left out. Do not anchor on any prior numbers; everything empirical starts with you.
* The owner's earlier brainstorming with ChatGPT is incorporated (§2, §3.2–§3.8, §3.16, §4; summary and corrections in Appendix D).
* If anything in this brief is wrong, fix it in the code and note it in `RESEARCH_LOG.md`. The math is checked, but the guidance on experiment design is judgment.

### 0.1 Mission (one paragraph)

SGD uses a single global learning rate for all parameters, plus a hand-designed schedule over time. Transformers have strongly **heterogeneous** curvature and gradient scales across parameter blocks, and **non-stationary** statistics over training, so one scalar cannot fit all blocks at all times. Adam's per-coordinate second-moment normalization is the usual fix, but it costs memory and is a blunt instrument. Your job is to build an SGD-family optimizer whose **per-block scalar learning rates $\alpha_k(t)$ are learned online** from a hypergradient computed in an **anchored reparameterization** of SGD ($\theta_t=\theta_0-\alpha_t S_t$, with $S_t$ the accumulated gradients). It should learn both the cross-block scales (which can differ by orders of magnitude) and the temporal profile (warmup and annealing), with SGD-level memory and minimal tuning. Then establish rigorously when and why it works, compared against strong, fairly tuned baselines.

### 0.2 Content for `CLAUDE.md` (create this file verbatim)

```markdown
# Project: learned per-block learning rates for SGD (anchored / horizon hypergradients)
Read RESEARCH_BRIEF.md fully before starting; re-read the relevant section before each phase.

Standing rules
1. Every number in a report cites run IDs from runs/registry.csv. No number without a run ID.
2. Equal tuning budget for baselines and our method. Pre-register each sweep (grid, seeds, budget)
   in RESEARCH_LOG.md BEFORE launching it.
3. Smoke-test (<= 3 min) the exact config before any run longer than 1 GPU-hour.
4. Optimizer state (S, log-alpha, q, momentum) is fp32; all block dot products accumulate in fp32.
5. Any new/fast optimizer implementation must match the reference implementation
   (RESEARCH_BRIEF.md Appendix A) to <= 1e-10 in float64 on the tests in Appendix B before use.
6. Use both GPUs: one job per GPU through scripts/queue.py unless a run explicitly needs DDP.
7. Never tune on a test set. Vision: carve a validation split from train; report test once.
8. Keep negative results. Never delete runs; mark them invalid with a reason instead.
9. Commit before launching; every run records the git SHA, full config, seed, and data order seed.
10. At each phase gate write reports/phase_<N>.md, update RESULTS.md, and send the owner a short summary.
11. Paired comparisons only: same data order, same token budget, same eval set across optimizers.
```

### 0.3 Autonomy and check-ins

* Work autonomously inside a phase. Stop at each **gate** (G0 to G5, §8), write the phase report, and message the owner.
* If the owner does not respond, take the **default branch** stated at that gate.
* Ask the owner once at the start whether they want Weights & Biases logging. If yes, request the API key; if not, use local JSONL plus TensorBoard.
* Ask before downloading ImageNet-scale data (the owner must provide access), before any single run expected to take more than 24 GPU-hours, and before pivoting the research question (§11.3).
* Keep `RESEARCH_LOG.md` as an append-only, dated lab notebook. Each entry records hypothesis → pre-registered design → result (with run IDs) → interpretation → next step.

---

## 1. The problem we are attacking

1. **Block heterogeneity.** Hessian spectra differ sharply across transformer parameter blocks (Q, K, V, attention output, MLP, embeddings, LayerNorm). On block-diagonal quadratics, GD's complexity scales with the *global* condition number $\kappa$, while Adam (β₁=0, β₂=1) scales with roughly $\max_l \kappa_l$ per block. The SGD–Adam gap tracks a measure of this heterogeneity (the Jensen–Shannon distance between block spectra, called JS⁰). CNNs are homogeneous and show little gap (Zhang et al., 2024, *Why Transformers Need Adam: A Hessian Perspective*). A related theory phrases the same issue as gradient heterogeneity (Tomihari & Sato, TMLR 2026).
2. **Class (token) imbalance in the output layer.** The gradient on an unembedding row scales with that token's frequency. Under GD, the loss of a rare class decays like $1/(\pi_k t)$ (Kunstner et al., 2024), so this calls for **per-row** scaling.
3. **Scale gap.** Adam's effective per-block learning rates are **2 to 4 orders of magnitude larger** than any LR plain SGD can sustain. SGD fails at such rates because of layer-wise gradient spikes and diverging rows for frequent tokens. *SGD-LL* (momentum SGD with layer-wise RMS clipping of updates plus per-token-class clipping in the output layer, base LR around 100–300) gets close to Adam. On GPT-2 124M on FineWeb it lands 7.78% above Adam, versus 49.70% for their unclipped SGD baseline. On C4 at 130M, 350M and 1B parameters (Chinchilla budgets) it reaches 3.24/2.87/2.60 versus Adam's 3.11/2.78/2.51 (Glentis et al., May 2026). **This is the strongest SGD-family competitor and must be a baseline.**
4. **Evidence that per-block scalars might suffice.**
   * Adam-mini: one second-moment scalar per Hessian-dense block (per head for Q/K, per output neuron for V/attn-proj/MLP, per token for embedding and output) matches AdamW.
   * Adalayer\*: per-layer Adam that treats the logit layer specially nearly recovers Adam; only the last layer and LayerNorm need adaptivity, and the other blocks can use SGD with ratios frozen at initialization (Zhao et al., 2024/25).
   * SGD-SaI: per-block scales computed at initialization from gradient signal-to-noise and then frozen; SGD-M with these scales is competitive on ViT and GPT-2.
5. **Non-stationarity.** The best LR changes over training (warmup, then decay). Schedules are hand-designed and tuned. Greedy (short-horizon) online LR adaptation is known to decay **too fast** (*short-horizon bias*; Wu et al., 2018).
6. **Batch size matters.** Vanilla SGD is competitive with Adam at small batch sizes, and the gap widens as batch size grows (Marek et al., 2025; Glentis et al., 2026).

**Target:** an SGD-family optimizer with ≤ 2 parameter-sized buffers (ideally 1), no hand-designed clipping rules, no tuned LR schedule, and low sensitivity to its initial LR. It should close most of the SGD-M → AdamW gap on transformers (and beat or match SGD-LL), while not hurting CNNs.

---

## 2. The idea on one page

**Reparameterization.** Plain SGD with a constant LR satisfies $\theta_t=\theta_0-\alpha\sum_{i<t}g_i=\theta_0-\alpha S_t$. Gradient descent is *not* invariant to reparameterization, so we make $\alpha$ a **trainable parameter of this map**, per block $k$:
$$\theta^{(k)}_t=\theta^{(k)}_0-\alpha^{(k)}_t S^{(k)}_t,\qquad S_{t+1}=S_t+g_t,\qquad g_t=\nabla L_t(\theta_t).$$
The derivative of the *current* minibatch loss with respect to $\alpha^{(k)}$, holding $S_t$ fixed, is $-\langle g^{(k)}_t,S^{(k)}_t\rangle$. That gives a persistent, learnable LR updated every step by a hypergradient that looks back over the **entire trajectory**.

**Equivalent views.** The same iterate is (i) SGD whose LR rescales the *whole* accumulated displacement, (ii) **blockwise dual averaging** with learned per-block regularization strengths, $\theta_t=\arg\min_w\{\langle S_t,w\rangle+\sum_k\tfrac{1}{2\alpha_k}\|w^{(k)}-\theta_0^{(k)}\|^2\}$ (§3.3b), and (iii) SGD plus a learned rank-one preconditioner along the accumulated direction (§3.4). A good one-line description of the method is **"hypergradient-adapted blockwise dual averaging"**.

**The owner's two original motivations**, and what the analysis says about each:
1. *Adaptive per-tensor/layer learning rates for SGD.* This is the research goal.
2. *Never update the weight tensors explicitly*: keep the anchor fixed, update only $S$ and $\alpha$, and evaluate $L(\theta_0-\alpha_tS_t)$. This is a valid implementation, but it is **not a computational saving**. The effective weights must still be materialized (or each layer computed as $Ax-\alpha(Sx)$, i.e. two matmuls), and the state (anchor + $S$ + α) is at least as large as SGD's. Do not claim efficiency from it. What the parameterization *does* buy is a learning rate that is differentiable through the **current** loss (§3.2).

**Family.** Replace the fixed anchor $\theta_0$ with an exponential moving average (EMA) of past iterates, with decay $\beta$. This yields one family indexed by the **hypergradient horizon** $\approx 1/(1-\beta)$ and a binary **retroactive-move** switch $r$:

* β = 0 is exactly Baydin et al.'s hypergradient descent (HD).
* β = 1 with r = 1 is the original anchored proposal.
* A variant with an exact curvature correction (one Hessian-vector product, or HVP, per step) gives each eigendirection its own horizon. The owner's one-step HVP formula is the first term of that correction.

**Canonical per-block step** (working name *HorizonHD*). For each block $k$, with direction $d_t=g_t$ or a momentum buffer:
$$h_t=\langle g_t,S_t\rangle,\quad \log\alpha_{t+1}=\Pi_{[\log\alpha_{\min},\log\alpha_{\max}]}\!\Big(\log\alpha_t+\mathrm{clip}_c\big(\mu\,\tfrac{h_t}{\|g_t\|\|S_t\|}\big)\Big),$$
$$\theta_{t+1}=\theta_t-\alpha_{t+1}d_t-r\,\beta\,(\alpha_{t+1}-\alpha_t)\,S_t,\qquad S_{t+1}=\beta S_t+d_t .$$
Here $\Pi$ projects $\log\alpha$ onto the bounds and $\mathrm{clip}_c$ limits the per-step change to $c$.

**Starting configuration for bring-up** (not a recommendation; resolve with Phase 3–4 data): per-tensor blocks plus per-row blocks for the embedding and unembedding matrices with relative bounds; β = 0.99; r = 1; cosine normalization; μ = 0.03; c = 0.05; heavy-ball momentum 0.9; global grad clipping at 1.0; weight decay 0; α₀ = 0.03 (with momentum) or 0.1 (without); bounds [1e-6, 1e3].

**Naming.** "AdaSGD" is taken. The working names here are *HorizonHD* (family), *AnchoredHD* (β = 1) and *SensitivityHD* (exact curvature correction). The owner will pick the final name.

---

## 3. Theory and derivations

### 3.1 Notation

| symbol | meaning |
|---|---|
| $k=1..K$ | blocks: a partition of the parameters (default: one PyTorch tensor each; see §3.13) |
| $\theta^{(k)}\in\mathbb R^{d_k}$, $x^{(k)}$ | parameters of block $k$; the block-$k$ restriction of any vector $x$ |
| $L_t$ | minibatch loss on batch $B_t$ (batches independent across $t$); $f=\mathbb E L_t$ |
| $g_t=\nabla L_t(\theta_t)$, $H_t=\nabla^2L_t(\theta_t)$ | minibatch gradient and Hessian at the current iterate (same batch) |
| $\alpha^{(k)}_t>0$; $A_t=\mathrm{blockdiag}(\alpha^{(k)}_tI_{d_k})$ | per-block learning rate; its block-diagonal matrix |
| $S_t$ | accumulated directions: $S_0=0$, $S_{t+1}=\beta S_t+d_t$ ($\beta=1$, $d=g$: $S_t=\sum_{i<t}g_i$) |
| $h^{(k)}_t=\langle g^{(k)}_t,S^{(k)}_t\rangle$ | first-order anchored hypergradient signal ($=-\partial L_t/\partial\alpha^{(k)}$) |
| $\mu$ | meta step size. **In the owner's boxed preconditioner formula this was called β; here β is the horizon.** |
| $\beta\in[0,1]$; $r\in\{0,1\}$ | horizon parameter (horizon ≈ $1/(1-\beta)$ steps); retroactive-move switch |
| $C$ | covariance of the minibatch-gradient noise |

### 3.2 The anchored reparameterization and its first-order hypergradient

Holding $S_t$ fixed (detached), differentiate the current loss through $\theta^{(k)}_t=\theta^{(k)}_0-\alpha^{(k)}S^{(k)}_t$:
$$\frac{\partial L_t}{\partial\alpha^{(k)}}=\Big\langle g^{(k)}_t,\frac{\partial\theta^{(k)}_t}{\partial\alpha^{(k)}}\Big\rangle=-\langle g^{(k)}_t,S^{(k)}_t\rangle=-h^{(k)}_t .$$
Gradient descent on $\alpha$ gives the original proposal (AnchoredHD, v1, raw form):
$$\alpha^{(k)}_{t+1}=\alpha^{(k)}_t+\mu\,h^{(k)}_t,\qquad S_{t+1}=S_t+g_t,\qquad \theta^{(k)}_{t+1}=\theta^{(k)}_0-\alpha^{(k)}_{t+1}S^{(k)}_{t+1}.$$

* **Order matters.** $h_t$ must use the old $S_t$, which contains only $g_0,\dots,g_{t-1}$ (batches independent of $B_t$). That is what makes it an unbiased estimate of the fresh-data meta-gradient (§3.9).
* **Order matters, quantitatively.** If $S$ is updated first and the "hypergradient" is computed as $\langle g_t,S_t+g_t\rangle=h_t+\|g_t\|^2$, it gains an always-positive bias $\|g_t\|^2$ that is not the derivative of the forward pass that was evaluated (α would drift upward).
* **First step.** $S_0=0$, so $\theta_0$ does not depend on $\alpha_0$: the first hypergradient is exactly 0 and adaptation starts at step 1.
* **Positivity.** Either project with $\alpha\leftarrow\max(\alpha_{\min},\alpha)$ (note: **max**, not min), or parameterize $\alpha=e^{s}$ with $s_{t+1}=s_t+\mu\,\alpha_t h_t$ (chain rule: $\partial L/\partial s=\alpha\,\partial L/\partial\alpha$); softplus is another option. These change the adaptation dynamics and count as distinct variants.
* **Log-parameter reading.** $\partial L_t/\partial s_k=-\alpha_kh^{(k)}_t=\langle g^{(k)}_t,\theta^{(k)}_t-\theta^{(k)}_0\rangle$: it asks whether **expanding the block's current displacement from initialization** helps or hurts the current loss. Setting $\alpha_k=0$ (buffer fixed) resets the block to its initialization; setting an ordinary SGD LR to 0 merely freezes it.
* **Autograd view (exact, first order).** Per block, `w = anchor - alpha * S.detach()`; evaluate the model at `w` (e.g. `torch.func.functional_call`); one backward gives both $g_t=\nabla_wL$ and $\partial L/\partial\alpha=-\langle g_t,S_t\rangle$. With `anchor.requires_grad=True`, $\nabla_{\rm anchor}L=\nabla_wL$ exactly, because $\partial w/\partial\,{\rm anchor}=I$: there is no extra Jacobian scaling. (The owner's $f(aw+b)$ analogy does not apply: the anchor enters as $w=A+\text{const}$, like $z=w+b$, not $z=aw+b$. The only Jacobian scaling is for the buffer, $\partial w/\partial S=-\alpha I$; see §3.8.) A fused optimizer that computes $h$ from the ordinary weight gradient is equivalent and cheaper (Appendix A).
* **Why the retroactive parameterization is needed for a current-loss hypergradient.** The prospective alternative, $w_t=A-D_t$ with $D_{t+1}=D_t+\alpha_tg_t$, reproduces ordinary SGD with time-varying LRs, but then $\partial L_t(w_t)/\partial\alpha_t=0$ (the current weights do not depend on the new rate), so learning α requires differentiating a proposed *next* step (HD-style). The anchored form makes α differentiable from the current loss **precisely because** changing α changes the current weights. The retroactive move (§3.4) is the price of that property; whether it helps or hurts is a central empirical question (A1).

### 3.3 Lemma 1: closed form of α [Verified]

For the raw additive update, per block:
$$\|S_{i+1}\|^2=\|S_i\|^2+2\langle g_i,S_i\rangle+\|g_i\|^2\;\;\Rightarrow\;\;\sum_{i\le t}h_i=\tfrac12\Big(\|S_{t+1}\|^2-\sum_{i\le t}\|g_i\|^2\Big)=\sum_{0\le j<i\le t}\langle g_i,g_j\rangle ,$$
$$\boxed{\alpha_{t+1}=\alpha_0+\mu\sum_{j<i\le t}\langle g_i,g_j\rangle}\qquad\text{vs. HD:}\qquad \alpha^{\rm HD}_{t+1}=\alpha_0+\mu\sum_{i\le t}\langle g_i,g_{i-1}\rangle .$$

* **Reading.** The anchored α integrates the agreement of *all pairs* of past gradients; HD uses only *adjacent* pairs. α rises while the trajectory keeps moving consistently and falls when gradients cancel (oscillation or noise).
* **Consequence.** The raw signal grows like $t\|g\|^2$ and differs by orders of magnitude across blocks and over time, so a single raw μ cannot serve all blocks. Normalize (§3.7).

### 3.3b Dual-averaging interpretation (important prior work)

For positive block scales,
$$\theta_t=\arg\min_w\Big\{\sum_k\langle S^{(k)}_t,w^{(k)}\rangle+\sum_k\frac{1}{2\alpha_{k,t}}\|w^{(k)}-\theta^{(k)}_0\|^2\Big\}\;\Longrightarrow\;S^{(k)}_t+\tfrac{1}{\alpha_{k,t}}\big(w^{(k)}-\theta^{(k)}_0\big)=0\;\Longrightarrow\;\theta^{(k)}_t=\theta^{(k)}_0-\alpha_{k,t}S^{(k)}_t .$$
This is Euclidean dual averaging / FTRL on linearized losses (Nesterov 2009; Xiao 2010), with the per-block step $\alpha_k$ acting as an inverse regularization strength toward the anchor. Consequences:
* **Modernized dual averaging (MDA; Jelassi & Defazio, 2020)** uses $z_{k+1}=x_0-s_k/\beta_k$ with a *prescribed* global scale $\beta_k=\sqrt{k+1}$ (weighted sum $s_k=\sum_i\lambda_ig_i$, $\lambda_k=\eta_k\sqrt{k+1}$), plus iterate averaging as momentum. With an L2 mirror map it is equivalent to SGD on $f(x)+\tfrac{\alpha_k}{2}\|x-x_0\|^2$ with $\alpha_k\sim1/(k+2)$: a decaying pull toward initialization. **MDA matched Adam on IWSLT'14 and beat it on Wikitext-103 (31.5 vs 33.1 test perplexity), where SGD failed (78.5).** This is an existence proof that the *anchored* parameterization with a suitable global scale schedule already helps transformers. It is both a mandatory baseline and a key control: **do learned per-block scales beat a prescribed √k global scale?** MADGRAD (Defazio & Jelassi, JMLR 2022) is its adaptive, momentumized successor.
* **D-Adaptation** (Defazio & Mishchenko, 2023) writes its iterate as $x_{k+1}=x_0-\gamma_{k+1}s_{k+1}$, and its §4.1 explicitly calls $\langle g_k,s_k\rangle$ "the (negative) hyper-gradient", using it to lower-bound the distance to the solution rather than to take gradient steps on the scale.
* A decreasing $\alpha_k$ strengthens the pull toward the anchor (stronger regularization in dual-averaging terms; the retroactive contraction of §3.4 in θ-space). This is a natural lens for the flatness/implicit-regularization thread (H10).

### 3.4 θ-space form, Lemma 2 (PSD preconditioner) and the ray identity [Verified]

With β = 1:
$$\theta_{t+1}=\theta_0-\alpha_{t+1}(S_t+g_t)=\theta_t-\alpha_{t+1}g_t-(\alpha_{t+1}-\alpha_t)S_t .$$

**Lemma 2.** Suppose $\alpha_{t+1}-\alpha_t=\phi_t(h_t)$ with $\phi_t(h)\,h\ge0$. This holds for the raw, exp, cos, rms, natural, clipped and bounded updates in §3.7. Then
$$\theta_{t+1}=\theta_t-P_tg_t,\qquad P_t=\alpha_{t+1}I+\frac{\phi_t(h_t)}{h_t}S_tS_t^{\top}\succeq\alpha_{t+1}I .$$
For the raw update, $P_t=\alpha_{t+1}I+\mu S_tS_t^\top$, which is the owner's boxed $P_{k,t}$ (applied per block).

* **Always a descent direction.** $\langle g_t,P_tg_t\rangle\ge\alpha_{t+1}\|g_t\|^2$, so the step always decreases the current minibatch's linear model.
* **Sign-adaptive momentum along the total displacement.** The rank-one term pushes further along $-S_t$ when $\langle g_t,S_t\rangle>0$ and retracts toward $\theta_0$ when $\langle g_t,S_t\rangle<0$.

**Spectrum and stability of the effective preconditioner.** For the raw update, $P_t$ has eigenvalue $\alpha_{t+1}+\mu\|S_t\|^2$ along $S_t$ and $\alpha_{t+1}$ on its orthogonal complement. So although only one scalar per block is learned, the step treats the accumulated direction differently from the rest of the block: the update is **not** a pure block-scalar rescaling of $g_t$. For a locally $L$-smooth loss, a sufficient condition for the full step to decrease the *same* minibatch loss is $L\,\lambda_{\max}(P_t)<2$, i.e. $L(\alpha_{t+1}+\mu\|S_t\|^2)<2$. Linearizing the coupled raw dynamics on $\tfrac12\lambda(w-w^\star)^2$ around an equilibrium $(\alpha^\star,S^\star)$ gives $e_{t+1}\approx[1-\lambda(\alpha^\star+\mu(S^\star)^2)]e_t$, so the coupled method needs $0<\lambda(\alpha^\star+\mu(S^\star)^2)<2$: **the rank-one term can destabilize training even when $\alpha^\star$ alone is safe**, and the danger grows with $\|S\|^2$.

**The α-subproblem has its own curvature.** For one block with the buffer frozen, $\phi(\alpha)=L(\theta_0-\alpha S)$ has $\phi'(\alpha)=-S^\top g$ and $\phi''(\alpha)=S^\top HS$. Raw gradient descent on α is stable only if $0<\mu<2/(S^\top HS)$, a bound that moves as $S$ grows. With several blocks, the α-Hessian is $\partial^2F/\partial\alpha_k\partial\alpha_j=S^{(k)\top}H_{kj}S^{(j)}$: **the scalar problems are coupled through cross-block curvature.** Note that $S^\top HS/\|S\|^2$ is exactly the Rayleigh quotient along the displacement used in §3.11 and E3.2. With the normalized meta-step $\Delta\alpha=\gamma\,\langle g,S\rangle/(\|S\|^2+\varepsilon)$ (§3.7, "natural"), the extra preconditioner is $\gamma SS^\top/(\|S\|^2+\varepsilon)$, whose largest eigenvalue is $\le\gamma$; its scale no longer grows with $\|S\|^2$, and the stability condition along the ray becomes roughly $\gamma R<2$.

**Ray identity (R).** Substituting $S_t=(\theta_0-\theta_t)/\alpha_t$:
$$\boxed{\theta_{t+1}-\theta_0=\frac{\alpha_{t+1}}{\alpha_t}(\theta_t-\theta_0)-\alpha_{t+1}g_t}$$
Every meta-step rescales the **entire** displacement from $\theta_0$. The α-update is a one-gradient-step line search along the ray $\theta_0\to\theta_t$. The term $-(\alpha_{t+1}-\alpha_t)S_t$ is called the **retroactive move** below.

### 3.5 The horizon family (Lemma 3) [Verified]

Use an EMA anchor: $a_0=\theta_0$, $a_{t+1}=\beta a_t+(1-\beta)\theta_t$, $S_{t+1}=\beta S_t+g_t$, $\theta_t=a_t-\alpha_tS_t$. Then
$$\theta_{t+1}=\beta a_t+(1-\beta)\theta_t-\alpha_{t+1}(\beta S_t+g_t)$$
$$=\beta(\theta_t+\alpha_tS_t)+(1-\beta)\theta_t-\alpha_{t+1}\beta S_t-\alpha_{t+1}g_t$$
$$=\theta_t-\alpha_{t+1}g_t-\beta(\alpha_{t+1}-\alpha_t)S_t ,$$
and in displacement form, $\theta_{t+1}-a_{t+1}=\beta\frac{\alpha_{t+1}}{\alpha_t}(\theta_t-a_t)-\alpha_{t+1}g_t$.

**Lemma 3.** β = 0 gives $S_t=g_{t-1}$, $h_t=\langle g_t,g_{t-1}\rangle$ and $\theta_{t+1}=\theta_t-\alpha_{t+1}g_t$. That is exactly HD (Baydin et al., 2018; Almeida et al., 1998), with an identical trajectory. β = 1 is the anchored proposal.

The family has **two independent design axes**:
* **(a) Signal horizon β:** how far back the hypergradient looks (≈ $1/(1-\beta)$ steps).
* **(b) Retroactive move r:** $\theta_{t+1}=\theta_t-\alpha_{t+1}g_t-r\beta(\alpha_{t+1}-\alpha_t)S_t$. With r = 1, α parameterizes the (recent) path. With r = 0 ("decoupled"), α scales only future steps and $S$ is pure memory for the signal.

With constant α, every member reduces to plain SGD. They differ only in how α moves and whether that move is applied to the past.

### 3.6 Curvature-corrected hypergradients: the owner's v2, the exact recursion, per-block subtleties

**v2 (one-step, the owner's derivation) [Verified].** v1 treats $S_t$ as constant, but $S_t=S_{t-1}+g_{t-1}(\theta_0-\alpha S_{t-1})$ depends on α. Unrolling one step with $S_{t-1}$ detached:
$$\frac{d\theta_t}{d\alpha}=-S_t+\alpha H_{t-1}S_{t-1},\qquad\boxed{\frac{dL_t}{d\alpha}=-\langle g_t,S_t\rangle+\alpha\langle g_t,H_{t-1}S_{t-1}\rangle}\quad\text{(global α).}$$
$H_{t-1}$ is the Hessian of batch $B_{t-1}$ at $\theta_{t-1}$. Implementation tip: compute $w=H_{t-1}S_{t-1}$ at step $t-1$, right after $g_{t-1}$, on the same batch, and store it in one buffer.

* **Sign.** The correction is **$+\alpha H_{t-1}S_{t-1}$** in $d\theta_t/d\alpha$ (an early draft had $-$). Equivalent form: $\frac{d\theta_t}{d\alpha}=(\alpha H_{t-1}-I)S_{t-1}-g_{t-1}$.
* **What it is the exact derivative of.** It is exact for the one-step surrogate $\theta_t(\alpha)=\theta_0-\alpha[S_{t-1}+g_{t-1}(\theta_0-\alpha S_{t-1})]$ with $S_{t-1}$ detached and **one persistent α shared by both steps**. If $g_{t-1}$ was produced with an independent earlier scalar $\alpha_{t-1}$ and the current α is a fresh variable, then $\partial g_{t-1}/\partial\alpha_t=0$ and the correction vanishes. For ordinary time-indexed SGD, $\partial\theta_{t+1}/\partial\alpha_t=-g_t$ with no Hessian term. Define the scalar's identity across steps explicitly, in code and in the paper.
* **v1 and v2 optimize different objectives,** not a worse and a better estimate of one derivative. v1 is the exact gradient of $L_t(\theta_0-\alpha S_t)$ (frozen buffer); v2 is the exact gradient of the one-step surrogate. Neither is the derivative of the realized training trajectory, and the omitted older Hessian terms can reinforce or cancel the retained one. The truncation depth is itself a research axis: **Level 0** (v1, no HVP) → **Level 1** (v2, one HVP) → **Level K** (differentiate through K unrolled steps; memory grows with K) → **exact forward-mode recursion** (below; one HVP per step, no unrolling).
* **"Lookahead" vs "current loss" organization.** Evaluating v2 at step $t$ with $H_{t-1}$, or at step $t-1$ via a differentiable lookahead $\tilde\theta_t(\alpha)=\theta_{t-1}(\alpha)-\alpha g_{t-1}(\theta_{t-1}(\alpha))$ scored on the next batch, gives the same value with shifted indices. The asymptotic cost is the same, but naively retaining the previous step's double-backward graph across iterations raises peak memory (two iterations' activations alive at once). **Compute the needed HVP explicitly and detach its result** instead.
* **Do not substitute Hessian proxies** for $H$ here. Adam's $\sqrt{v}$ measures gradient second moments (diagonal, nonnegative), not curvature; Hutchinson estimates diagonals or traces. The method needs a specific Hessian-*vector* product, which autograd or a finite difference gives directly.
* **Diagnostics for whether curvature matters:** the per-block ratio $|\text{curvature term}|/|\langle g^{(k)}_t,S^{(k)}_t\rangle|$, and the fraction of steps on which the correction flips the sign of the α update (log both in A3).

**Exact (full-horizon forward mode, α held fixed over the past) [Verified].** Let $q_t:=dS_t/d\alpha$ with $q_0=0$:
$$\frac{d\theta_t}{d\alpha}=-(S_t+\alpha q_t),\qquad q_{t+1}=q_t+H_t\frac{d\theta_t}{d\alpha}=q_t-H_t(S_t+\alpha q_t),\qquad -\frac{dL_t}{d\alpha}=\langle g_t,\,S_t+\alpha q_t\rangle .$$
* v2 is this recursion truncated at one step ($q_t\approx-H_{t-1}S_{t-1}$).
* An optional forgetting factor gives $q_{t+1}=\rho\,(q_t-H_t(S_t+\alpha q_t))$.
* Cost: one HVP per step (on $S_t$ for v2, on $S_t+\alpha q_t$ for the exact form) plus one extra buffer $q$.

**Weights on past gradients.** Take a locally constant $H$ and an eigendirection with curvature $\lambda_i$, and write $-\frac{d\theta_t}{d\alpha}=\sum_{k\ge0}w_kg_{t-1-k}$:

| signal | $w_0$ | $w_k$ ($k\ge1$) | effective horizon |
|---|---|---|---|
| HD (β=0) | 1 | 0 | 1 step |
| leaky (β) | 1 | $\beta^k$ | $1/(1-\beta)$ |
| anchored v1 (β=1) | 1 | 1 | entire run |
| anchored v2 | 1 | $1-\alpha\lambda_i$ | entire run, uniformly damped |
| exact | 1 | $(1-\alpha\lambda_i)^k$ | $1/(\alpha\lambda_i)$, **per direction** |

* **The exact sensitivity gives each direction its own memory.** Sharp directions forget quickly and flat ones remember long. No single β can do this.
* **v2 only multiplies all older weights by $(1-\alpha\lambda_i)$.** [Prediction] It is therefore nearly inert when $\alpha\lambda\ll1$ (v2 ≈ v1). If curvature correction matters at all, the exact form is the one to test.
* **Equivalence.** With constant α, the retroactive (anchored) and decoupled trajectories are the same plain-SGD trajectory, so their exact derivatives coincide; only the truncations differ.
* **SMD form.** The non-retro exact form is Schraudolph's stochastic meta-descent (SMD) with one scalar gain per block: $v_{t+1}=\rho(v_t-\alpha_{t+1}H_tv_t)-g_t$, $h_t=-\langle g_t,v_t\rangle$, $\theta_{t+1}=\theta_t-\alpha_{t+1}g_t$. This is `SensitivityHD` in Appendix A.

**Lemma 4: per-block exactness [Verified].** With separate $\alpha_k$ and cross-block curvature, the single-HVP recursions above are exact only for a **global α** or a **block-diagonal Hessian**. The exact per-block v2 is
$$\frac{dL_t}{d\alpha_k}=-\langle g^{(k)}_t,S^{(k)}_t\rangle+\big\langle[H_{t-1}A\,g_t]^{(k)},\,S^{(k)}_{t-1}\big\rangle ,$$
which follows from the symmetry of $H$. It needs $H_{t-1}$ applied to a vector from step $t$: either store batch $t-1$ and a copy of $\theta_{t-1}$, or approximate $H_{t-1}\approx H_t$ on the current batch, which reintroduces same-batch correlation. Written out with $v^{(j)}=\alpha_jg^{(j)}_t$: $\partial L_t/\partial\alpha_k=-\langle g^{(k)}_t,S^{(k)}_t\rangle+\sum_j\alpha_j\langle g^{(j)}_t,H^{(j,k)}_{t-1}S^{(k)}_{t-1}\rangle=-\langle g^{(k)}_t,S^{(k)}_t\rangle+\langle S^{(k)}_{t-1},(H_{t-1}v)^{(k)}\rangle$. So **one full-network HVP gives all K exact per-block hypergradients**, and the cross-block terms capture how changing an early layer's rate changes later layers' gradients. A memory-light recipe: keep the indices of batch $t-1$ plus $g_{t-1}$ and $S_{t-1}$ (so $\theta_{t-1}=\theta_0-A_{t-1}S_{t-1}$ can be rebuilt for β = 1 without weight decay), and compute $H_{t-1}v\approx[\nabla L_{t-1}(\theta_{t-1}+\varepsilon v)-g_{t-1}]/\varepsilon$: one extra forward/backward on the previous batch, plus one buffer.
* The cheap form $\alpha_k\langle g^{(k)}_t,[H_{t-1}S_{t-1}]^{(k)}\rangle$ is the SMD-style diagonal approximation.
* On small random MLPs it differed from the exact value by roughly 10–60% per block (Appendix B, T5).
* Treat the cheap form as an approximation; if curvature correction turns out to matter, compare it against the exact form.

**Computing HVPs in transformers.**
* *Pearlmutter double-backward* needs `create_graph=True`. Flash and memory-efficient SDPA kernels do **not** support double backward, so use the math SDPA backend for HVP passes (memory-heavy).
* *Finite difference:* $Hv\approx[\nabla L_t(\theta_t+\varepsilon v)-\nabla L_t(\theta_t)]/\varepsilon$ costs one extra forward/backward (same batch, same dropout mask); a central difference costs two. Pick $\varepsilon$ so that $\varepsilon\|v\|\approx10^{-3}\|\theta\|$, run HVP passes with fp32 accumulation, and validate against autograd on a small model with math attention (Appendix B).
* *Gauss–Newton vector products* are a PSD alternative (SMD used them).
* *Amortization:* compute the HVP every $m$ steps.

### 3.7 The meta-step: parameterization and normalization

The raw $h$ has units of loss per unit α and spans orders of magnitude, so normalization is needed. Every option below is sign-preserving, so Lemma 2 holds.

1. **raw:** $\alpha\mathrel{+}=\mu h$, projected.
2. **exp:** $\log\alpha\mathrel{+}=\mu\,\alpha h$.
3. **cos (default):** Baydin's multiplicative HD form, $\log\alpha\mathrel{+}=\mu\,\cos(g,u)$ with $u$ the sensitivity vector ($S$ for v1). Scale-free, with $|\Delta\log\alpha|\le\mu$.
4. **rms:** $\log\alpha\mathrel{+}=\mu\,(\alpha h)/\mathrm{RMS}_{\rm EMA}(\alpha h)$, i.e. Adam applied to $\log\alpha$.
5. **natural:** use the metric $\|\partial\theta/\partial\alpha\|^2=\|S\|^2$, giving $\Delta\alpha=\mu h/\|S\|^2$. The retroactive move then becomes $-\mu\hat S\hat S^\top g$: a gradient step of size μ along the unit displacement direction ($P=\alpha I+\mu\hat S\hat S^\top$). Here μ is literally a learning rate for the "ray coordinate".
6. **coin-betting tuner per block** (Mechanic-style, parameter-free).

Always apply a per-step clip $|\Delta\log\alpha|\le c$ and bounds $[\alpha_{\min},\alpha_{\max}]$. For row blocks, set the bounds relative to the parent tensor's α.

**Runaway caveat (mechanism).** The scale-free cos update grows α exponentially when a block's gradient keeps a consistent direction while the loss has no minimum along it. Example: unembedding rows of tokens that never occur, whose logits softmax keeps pushing down. Use bounds, skip the meta-update for rows with negligible gradient, or shrink row-level α toward the tensor-level α.

### 3.8 Updating S by gradient descent instead of accumulation (the owner's open question)

In $\theta=\theta_0-\alpha S$ we have $\partial L/\partial S=-\alpha g$ and $\partial L/\partial\alpha=-\langle g,S\rangle$. Gradient descent on both gives $S_{t+1}=S_t+\eta\,\alpha_tg_t$ and $\alpha_{t+1}=\alpha_t+\mu\langle g_t,S_t\rangle$.

* Accumulation ($S\mathrel{+}=g$) **is** gradient descent on $S$ with $\eta_t=1/\alpha_t$. The default is already a special case.
* With a fixed η: $\theta_{t+1}=\theta_t-\eta\,\alpha_t\alpha_{t+1}g_t-(\alpha_{t+1}-\alpha_t)S_t$. The effective LR is $\eta\alpha^2$: quadratic in α and self-reinforcing.
* **Lemma 5 (balancedness) [Verified].** Under gradient flow, $\tfrac{d}{d\tau}\big(\alpha^2/\mu-\|S\|^2/\eta\big)=2\alpha\langle g,S\rangle-2\alpha\langle g,S\rangle=0$. The scalar × vector factorization behaves like a two-layer linear model, so the balancedness and implicit-bias results for factorized parameterizations apply (Du, Hu & Lee, 2018).

* **Accumulation is preconditioned GD on the buffer.** Since $\nabla_{S}L_t=-\alpha_tg_t$, the rule $S_{t+1}=S_t+g_t$ is exactly $S_{t+1}=S_t-\frac{1}{\alpha_t}\nabla_SL_t$: gradient descent on $S$ in the metric $\alpha_tI$ (blockwise, step $1/\alpha_{k,t}$). It is the parameterization-aware choice that keeps the SGD interpretation at fixed α, which is the cleanest justification for the default.
* **Effective preconditioners.** Accumulation: $P^A_t=\alpha_{t+1}I+\mu S_tS_t^\top$. Euclidean GD on $S$: $P^B_t=\gamma\alpha_t\alpha_{t+1}I+\mu S_tS_t^\top$. Same rank-one correction, different isotropic part. With $\gamma=1$ and $\alpha=0.01$ the fixed-α effective LR is $10^{-4}$, not $10^{-2}$. On $\tfrac12\lambda e^2$ the fixed-α stability conditions are $\alpha\lambda<2$ versus $\gamma\alpha^2\lambda<2$, so with one γ shared across blocks whose α differ by orders of magnitude, variant B updates some blocks far too slowly and others too aggressively.
* **Ablation family:** $S^+=S+\gamma\alpha^pg$ ($p=0,\gamma=1$: accumulation; $p=1$: Euclidean GD; $\gamma=1/\alpha$ with $p=1$ recovers accumulation exactly). **Always compare variants at matched effective weight-space LR**, not matched numerical α.

**Recommendation:** keep accumulation as the default. Test fixed-η gradient descent on $S$ only as a bounded exploratory ablation. The conservation law is a theory thread (implicit bias and flatness).

### 3.9 HD's "lookahead" is not a flaw (Lemma 6) [Verified]

The concern: HD differentiates $L_{t+1}(\theta_{t+1})$ (a fresh batch), while the math seems to call for $L_t(\theta_{t+1})$. The resolution is that the meta-objective is the **expected** loss $\mathbb E_B[L_B(\theta_{t+1}(\alpha))]$.

Take a quadratic $f$ and $L_B(\theta)=f(\theta)+\langle\xi_B,\theta\rangle$ with $\mathrm{Cov}(\xi)=C$, and let $\theta'=\theta-\alpha g_B$:
$$\mathbb E\Big[-\tfrac{d}{d\alpha}L_B(\theta')\Big]=\|\nabla f\|^2+\mathrm{tr}C-\alpha\big(\nabla f^\top H\nabla f+\mathrm{tr}(HC)\big)\quad\text{(same batch)}$$
$$\mathbb E\Big[-\tfrac{d}{d\alpha}L_{B'}(\theta')\Big]=\|\nabla f\|^2-\alpha\big(\nabla f^\top H\nabla f+\mathrm{tr}(HC)\big)\quad\text{(fresh batch)}$$
$$\Rightarrow\;\alpha^*_{\rm same}=\frac{\|\nabla f\|^2+\mathrm{tr}C}{\nabla f^\top H\nabla f+\mathrm{tr}(HC)},\qquad \alpha^*_{\rm fresh}=\frac{\|\nabla f\|^2}{\nabla f^\top H\nabla f+\mathrm{tr}(HC)} .$$
* $\alpha^*_{\rm fresh}$ is exactly McCandlish et al.'s noise-aware optimal step $\epsilon_{\max}/(1+B_{\rm noise}/B)$.
* Near the optimum, $\alpha^*_{\rm fresh}\to0$ (it anneals), while $\alpha^*_{\rm same}\to\mathrm{tr}C/\mathrm{tr}(HC)$ and never anneals, because the batch's own noise is counted as progress.

So **HD's lookahead is the correct estimator.** The anchored $h_t=\langle g_t,S_t\rangle$ also pairs a fresh $g_t$ with older gradients. **Do not present the "lookahead fix" as a contribution.** The real differentiators are persistence with a long horizon, the reparameterization (retroactive move and curvature corrections), and per-block granularity.

### 3.10 The noise floor drives every member to anneal (Lemma 7) [Verified by simulation]

Consider constant-α SGD at stationarity on a quadratic, in one eigendirection with curvature λ and noise variance $c$. The error obeys $\delta_{t+1}=(1-\alpha\lambda)\delta_t-\alpha\xi_t$, so $\mathbb E\delta^2=\frac{\alpha c}{\lambda(2-\alpha\lambda)}$. For lag $k\ge1$, $\mathbb E[g_tg_{t-k}]=-(1-\alpha\lambda)^{k-1}\frac{\alpha\lambda c}{2-\alpha\lambda}$. Weighting lags by $\beta^{k-1}$ and summing over directions:
$$\boxed{\mathbb E[h_t]=-\sum_i\frac{\alpha\lambda_ic_i}{(2-\alpha\lambda_i)\,(1-\beta(1-\alpha\lambda_i))}}\qquad \beta=0:\;\approx-\tfrac{\alpha}{2}\mathrm{tr}(HC);\qquad \beta=1:\;=-\sum_i\tfrac{c_i}{2-\alpha\lambda_i}\approx-\tfrac12\mathrm{tr}C .$$

* **Annealing emerges without a schedule.** At the noise floor, every member pushes α down.
* **The horizon decides which directions vote.** With β = 0, direction $i$ is weighted by $\lambda_i$, so sharp directions dominate and α anneals to suit them. This is a mechanistic explanation of short-horizon bias.
* With β → 1, every direction is weighted equally (≈ $c_i/2$). [Prediction] Long horizons therefore anneal less aggressively and protect slow (flat) directions.

### 3.11 β = 1 with the retroactive move: two predicted failure modes and one counter-prediction

* **P1, dilution [Prediction].** At the noise floor, $\cos(g_t,S_t)\approx\mathbb E[h]/(\|g\|\|S\|)\approx-\tfrac{\alpha}{2}\,\|g\|/\|\theta_t-\theta_0\|$ (using $\|S\|=\|\theta_t-\theta_0\|/\alpha$ and $\|g\|^2\approx\mathrm{tr}C$). The annealing signal shrinks as the iterate travels, so a large μ is needed to anneal.
* **P2, amplification [Prediction].** By (R), $\delta\theta=(\delta\alpha/\alpha)(\theta_t-\theta_0)$: jitter in α moves θ along the *whole* displacement, so a large μ is unaffordable.
* **Together** these predict that β = 1 with r = 1 gets stuck with a small μ and a frozen α (no annealing). The EMA anchor fixes both, because it rescales only the recent path (displacement form in §3.5). Setting r = 0 fixes P2.
* **Counter-prediction (why it might be fine in deep nets).** The loss cost of a ray move is $\Delta L\approx\frac{\delta\alpha}{\alpha}\langle g,D\rangle+\frac12\big(\frac{\delta\alpha}{\alpha}\big)^2D^\top HD$ with $D=\theta_t-\theta_0$. Monotonic linear interpolation (MLI; Lucas et al., 2021) suggests the loss along $\theta_0\to\theta_t$ is often smooth and monotone. If the per-block Rayleigh quotient $R^{(k)}_t=D^{(k)\top}H D^{(k)}/\|D^{(k)}\|^2\ll\lambda^{(k)}_{\max}$, ray moves are cheap. **Measure it (E3.2); it decides the β = 1 question.**
* **Possible upside.** When α decreases, the retroactive move pulls θ toward the anchor: a Lookahead-style interpolation toward slow weights (Zhang et al., 2019). This is a flatness and implicit-regularization thread.

### 3.12 Equivalences and the prior-work map

* **Lemma 8: KT coin betting ≡ anchored with a specific tuner [Verified].** The Krichevsky–Trofimov (KT) bettor in Hilbert space (Orabona & Pál, 2016), anchored at $\theta_0$, with rewards $c_t=-g_t$ ($\|g\|\le1$) and initial wealth ε, predicts $\theta_t=\theta_0-\alpha_tS_t$ with $\alpha_t=\mathrm{Wealth}_{t-1}/t$ and $\mathrm{Wealth}_t=\mathrm{Wealth}_{t-1}+\alpha_t\langle g_t,S_t\rangle$. Equivalently $\alpha_{t+1}=\alpha_t\frac{t+h_t}{t+1}$ with $\alpha_1=\varepsilon$. KT is thus a multiplicative anchored hypergradient with a $1/t$ decay, and it carries a parameter-free regret bound $O\big(\|u\|\sqrt{T\log(1+T\|u\|/\varepsilon)}\big)$.
* **Mechanic** (Cutkosky, Defazio & Mehta, 2023): $x_t=x_{\rm ref}+s_t\Delta_t$, with $\Delta$ the accumulated base-optimizer updates (for an SGD base, $\Delta=-S$). It feeds $h=\langle\Delta,g+\text{wd term}\rangle$ to a coin-betting tuner whose output sums six tuners with different β's. It is a **global** scalar; the authors list per-layer scales as future work. **The anchored parameterization is therefore not new. Per-block Mechanic is a mandatory baseline.**
* **D-Adaptation / Prodigy:** the numerator $\sum_i\lambda_i\langle g_i,x_0-x_i\rangle$ is a weighted cumulative anchored hypergradient, used to estimate the distance to the solution. **DoG / L-DoG:** $\eta=\max_i\|x_i-x_0\|/\sqrt{\sum\|g_i\|^2}$, an anchored distance with a per-layer variant.
* **HD / HDM** (Baydin et al., 2018; Chu et al., ICML 2025): β = 0 member. Chu et al. give theory in the deterministic convex setting only, and report HD warm-up instability (fixed with null steps and AdaGrad meta-steps).
* **SMD / IDBD:** exact sensitivity with per-coordinate gains; `SensitivityHD` uses one scalar gain per block.
* **GALA** (Jiang, Kavis & Mokhtari, 2025): alignment of consecutive gradients, with FTRL; a global scalar; ResNets only.
* **Single-buffer heavy-ball variant (memory equal to SGD-M):** $m_{t+1}=\beta_mm_t+g_t$, $\theta_{t+1}=\theta_t-\alpha_{t+1}m_{t+1}$, one-step hypergradient $h_t=\langle g_t,m_t\rangle$ (since $\partial\theta_t/\partial\alpha\approx-m_t$). Its horizon is tied to the momentum, ≈ $1/(1-\beta_m)$.

### 3.13 Granularity of blocks

| granularity | notes |
|---|---|
| global ($K=1$) | the Mechanic / HD setting; cannot address heterogeneity |
| per type (emb, head, LN/bias, attn, MLP) | cheap sanity point |
| **per tensor** (default) | matches PyTorch params; Zhang et al.'s block heterogeneity is at roughly this level |
| Hessian sub-blocks (Adam-mini partition) | per head for Q/K; per output neuron for V, attn-proj and MLP; per token row for embedding and output |
| per row | for embedding and unembedding (class imbalance) |
| per coordinate | IDBD/SMD territory; noisy; loses the block premise |

* **Trade-off.** Finer blocks fit heterogeneity better but give a noisier $h$ (SNR grows roughly with $\sqrt{d_k}$) and more runaway risk.
* **Mitigation:** hierarchical log-rates, $\log\alpha_{\rm row}=\log\alpha_{\rm tensor}+\delta_{\rm row}$, with δ updated using a smaller μ and shrunk toward 0.
* **Expectation from the literature:** per-tensor rates for hidden matrices plus per-row rates for the embedding and unembedding, with separate handling of LayerNorm and biases. Phase 2 decides this.

### 3.14 Momentum

* **Direction.** Use $d_t=m_t$ (heavy ball or Nesterov) with $S$ accumulating $d$. The signal stays $h=\langle g_t,S_t\rangle$, which is the exact one-step derivative.
* **Memory.** $S$ plus $m$ is two buffers, the same as AdamW. The single-buffer heavy-ball variant (§3.12) gives memory parity with SGD-M.
* **β = 1 storage.** With r = 1 and no weight decay, β = 1 can store $\theta_0$ instead of $S$, since $S_t=(\theta_0-\theta_t)/\alpha_t$ (this also holds with momentum when $S$ accumulates $d$). Memory is the same, but the identity is exact.

### 3.15 Per-block edge of stability (hypothesis H5)

On a deterministic quadratic with HD, each eigendirection obeys $g_t=(1-\alpha\lambda)g_{t-1}$, so
$$h_t=\sum_i\lambda_i^2\delta_i^2(1-\alpha\lambda_i).$$
* As fast directions converge, slow ones dominate the sum and α grows.
* Once $\alpha\lambda_{\max}>2$, the top direction diverges and its gradient alternates sign, which makes $h$ negative and cuts α.

[Prediction] So α is driven toward the stability edge, $\alpha\lambda_{\max}\in[1,2]$. This connects to edge-of-stability dynamics (EoS; Cohen et al., 2021). Per block, the prediction is $\alpha_k\lambda^{(k)}_{\max}\approx1$–$2$ early in training, then lower once noise dominates (§3.10). If true, the learned per-block α directly caps per-block sharpness ($\lambda^{(k)}_{\max}\lesssim2/\alpha_k$), which links the method to flat minima and to the owner's research interests.

### 3.16 Choice of outer objective (training vs held-out loss)

Everything above uses the training minibatch loss as the meta-objective. Alternatively, optimize α on a **held-out** loss, as a bilevel problem $\min_\alpha L_{\rm val}(\theta(\alpha))$. For one prospective step $\theta_{t+1}(\alpha)=\theta_t-\alpha\nabla L_{\rm train}(\theta_t)$, per block,
$$\frac{\partial L_{\rm val}(\theta_{t+1})}{\partial\alpha_l}=-\big\langle\nabla_{\theta_l}L_{\rm val}(\theta_{t+1}),\nabla_{\theta_l}L_{\rm train}(\theta_t)\big\rangle :$$
increase a block's LR when its training gradient also improves held-out loss. The anchored analogue uses the current point: $h^{\rm val}_t=\langle\nabla L_{\rm val}(\theta_t),S_t\rangle$. It costs one extra forward/backward on a held-out batch per meta-step (amortizable). This is related to MARTHE (Donini et al., 2020) and classical gradient-based hyperparameter optimization. It directly targets generalization (the owner's interest). Test it as an exploratory arm (A12), with the held-out split disjoint from the final evaluation set.

---

## 4. Related work and novelty positioning

| method | what is adapted | signal | horizon | granularity | extra memory |
|---|---|---|---|---|---|
| HD / HDM (Baydin 2018; Chu 2025) | LR (or preconditioner) | ⟨g_t, g_{t−1}⟩ | 1 step | global (diagonal in HDM-Best) | 1 buffer |
| Mechanic (2023) | scale s of the accumulated base updates | ⟨Δ, g⟩, coin betting | entire run | **global** | 1–2 buffers |
| KT coin betting (2016) | betting fraction | ⟨g, S⟩ | entire run | global | 1 buffer |
| D-Adapt / Prodigy | distance estimate d | Σλ⟨g, x₀−x⟩ | entire run | global (+ manual layer_scale) | 1–3 buffers |
| DoG / L-DoG | LR = distance / grad-norm | ‖x−x₀‖ | entire run | global / **per layer** | 1 buffer |
| Dual averaging / MDA (Xiao 2010; Jelassi & Defazio 2020) | nothing learned: prescribed global scale ∝ 1/√k | none (schedule) | entire run | global | 1–2 buffers |
| MADGRAD (2022) | dual averaging + cube-root adaptivity + momentum | Σλg² | entire run | per coordinate | 2–3 buffers |
| MARTHE (2020) | LR schedule | forward-mode sensitivity with discount | tunable | global | 1 buffer + HVP |
| SMD / IDBD | per-coordinate gains | exact sensitivity (HVP) | per direction | per coordinate | 2 buffers + HVP |
| GALA (2025) | LR | consecutive alignment + FTRL | short | global | — |
| SGD-SaI (2024) | per-block scale, **frozen at init** | gradient SNR at step 0 | — | per tensor | 0 |
| SGD-LL (2026) | update clipping (layer RMS + per-class) | norms | — | per layer / per class | 0 |
| Adam-mini (2024) | per-block Adam second moment | E[g²] | EMA | Hessian blocks | ~1 buffer |
| LANTON (2025) | per-layer LR on a Muon-like base | gradient-difference noise | EMA | per layer | — |
| HT-SR layerwise LR (2026) | per-layer LR on AdamW/Muon | weight spectra (power-law fit) | periodic | per layer | — |
| SGG (2025) | per-cluster LR multipliers (wrapper) | momentum deviation | EMA | clusters | small |
| Grafting (Agarwal 2020) | magnitude from optimizer A, direction from B | — | — | per layer | A's states |
| **Ours (HorizonHD family)** | **per-block LR (scale + schedule)** | **⟨g, S⟩ / exact sensitivity** | **tunable β, or per direction** | **tensor / row / Hessian block** | **1–2 buffers** |

**Honest novelty assessment.**
* The anchored parameterization and the $\langle g,S\rangle$ signal already exist (dual averaging, MDA, KT, Mechanic, and D-Adapt, whose §4.1 even names $\langle g_k,s_k\rangle$ the negative hypergradient). HD-type LR learning exists, and so does trajectory-sensitivity LR learning (MARTHE). Per-layer LRs for SGD exist in frozen (SGD-SaI) and clipping (SGD-LL) forms.
* **Defensible contributions:**
  * **(C1)** A unifying **horizon family** that separates *signal horizon* from the *retroactive move*, with exact equivalences (HD, KT/Mechanic, SMD), the PSD-preconditioner lemma, the noise-floor drift law (a mechanism for short-horizon bias and automatic annealing), and the dilution/amplification analysis predicting when anchor rays hurt.
  * **(C2)** A practical **per-block online LR learner for SGD**, aligned with Hessian blocks, that learns orders-of-magnitude scale differences *and* the schedule at SGD-level memory. It must match or beat SGD-LL and per-block Mechanic, and approach AdamW, without hand-designed clipping.
  * **(C3)** Mechanistic evidence: learned $\alpha_k$ versus per-block curvature (per-block edge of stability), versus Adam's effective LR, and the granularity actually required.
  * **(C4)** The flatness and generalization consequences of per-block α (the owner's core interest).
* **Novelty risks.** (o) MDA's prescribed global √k scale performs as well as learned per-block scales. (i) Per-block Mechanic or L-DoG performs as well. (ii) SGD-LL already closes the gap. (iii) Muon or Adam-mini dominate anyway. Mitigate by including them as baselines and by framing the work as C1 + C3 + C4 if C2 alone is not decisive.
* **Re-run a literature check at the start of Phase 0 and before writing.** Search (2026+) for "per-layer Mechanic", "layer-wise parameter-free SGD transformer", "hypergradient LLM pretraining" and "learning rate as parameter reparameterization". Keep `related_work.md` updated.

---

## 5. Known gaps and risks (critical review) and what to do about each

1. **Novelty overlap** (§4). Mandatory baselines: per-block Mechanic (on SGD-M), L-DoG, per-block HD, SGD-LL.
2. **The lookahead claim is not a contribution** (§3.9). Never claim it.
3. **β = 1 retroactive dynamics** may freeze α (P1/P2). Resolve with E3.2 diagnostics and the β × r ablation.
4. **Raw meta-steps are not scale-invariant.** Use normalized log-space updates; report raw only as an ablation.
5. **The per-block v2 cheap form is approximate** (Lemma 4). If HVP variants matter, compare the exact and cheap forms.
6. **HVP cost and flash-attention incompatibility.** Use finite-difference HVPs, amortize them, and report wall-clock as well as steps.
7. **Memory parity.** With momentum, the method uses 2 buffers (AdamW-level). Test the single-buffer heavy-ball variant and frame claims honestly.
8. **Unembedding-row runaway** (§3.7). Use bounds, shrinkage, or skip rows with tiny gradients.
9. **Interactions with weight decay, gradient clipping, warmup and external schedules.** Decide and document each one (§12) and ablate them (Phase 4).
10. **"LR-free" claims** require robustness of α₀ and μ across tasks without re-tuning (A7). Do not claim LR-free if μ is tuned per task.
11. **No stochastic non-convex guarantee yet.** With bounded log-steps and box bounds, standard SGD convergence with $\alpha_t\in[\alpha_{\min},\alpha_{\max}]$ holds trivially. A meaningful guarantee (regret against the best fixed per-block α in hindsight, in the stochastic setting) is open; see §14 theory tasks.
12. **"No explicit weight update" is not an efficiency gain** (§2). Never claim it; report step time and memory honestly.
13. **v2's semantics** (§3.6): exact only for a persistent α shared across the two steps; state this precisely, and label cheap block-diagonal approximations as approximations.
14. **Proxy-to-scale transfer.** Validate key conclusions at 124M, not only on proxies.

---

## 6. Research questions and pre-registered hypotheses

**RQ1 Ceiling.** Can per-block scalar LRs close the SGD-M → AdamW gap at all, and at what granularity? (Oracle experiments.)
**RQ2 Learnability.** Can these scales be learned online by HorizonHD, robustly?
**RQ3 Schedule.** Does the method learn the temporal profile (warmup and annealing) well enough to replace a tuned schedule?
**RQ4 Design.** Which choices matter: horizon β, retroactive move r, normalization, curvature correction, granularity, momentum?
**RQ5 Mechanism.** What do the learned α's track: curvature (edge of stability), Adam's effective LR, gradient SNR?
**RQ6 Generalization.** Effects on sharpness, flatness and test performance (vision especially).

| ID | hypothesis | test | falsified if |
|---|---|---|---|
| H1 | Oracle per-block scalars (grafted Adam magnitudes with the SGD-M direction) close ≥ 80% of the SGD-M→AdamW gap on the proxy LM; per-tensor alone may fall short for the embedding and output layers | E2.3–E2.5 | per-tensor plus per-row grafting closes < 50% |
| H2 | HorizonHD reaches ≥ 70% of the oracle's improvement | E3/E4 vs E2 | < 40% |
| H3 | Final loss improves with the horizon from β = 0 up to β* ∈ [0.99, 0.999] (short-horizon bias, §3.10) | A1 | β = 0 is best or flat across β |
| H4 | β = 1 with r = 1 underperforms β < 1 when the Rayleigh quotient along D is large; harmless when $R\ll\lambda_{\max}$ | A1 + E3.2 | no relation between R and the retroactive-move penalty |
| H5 | $\alpha_k\lambda^{(k)}_{\max}$ stays in [1, 2] early, then drops (per-block edge of stability) | E3.1 | product spreads over orders of magnitude |
| H6 | Without an external schedule, final loss is within 1% of the same method with a tuned cosine or WSD (warmup-stable-decay) schedule | A6 | worse by > 3% |
| H7 | Final loss stays within 2% of best across α₀ ∈ [best/30, best×30] | A7 | > 5% degradation within ×10 |
| H8 | Exact sensitivity beats v1 per step but not per wall-clock; v2 ≈ v1 | A3 | v2 clearly better than v1 |
| H9 | Gains on ResNet/CIFAR are small (homogeneous blocks) | E5 vision | large gains on CNNs (interesting, but contradicts the premise) |
| H10 | Learned per-block α yields lower per-block sharpness than AdamW at matched loss, and a smaller generalization gap on ViT/CIFAR | Phase 6 | no difference or the opposite |
| H11 | The advantage over SGD-M grows with batch size | A10 | shrinks |
| H12 | Learned per-block scales beat the same anchored form with a prescribed global scale (MDA, $\propto1/\sqrt{t+1}$, tuned constant) | E2.8 vs A4 | MDA matches or wins |
| H13 | A held-out-loss hypergradient (§3.16) gives a smaller generalization gap than the training-loss hypergradient at similar training loss | A12 | no difference |

Hypotheses H3, H4, H5 and H10 come from the theory in §3. Report all of them, confirmed or not.

---

## 7. Infrastructure

### 7.1 Hardware facts and implications
* 2× RTX 5080, **16 GB each**, Blackwell consumer (sm_120), BF16 tensor cores. **No NVLink**, so cross-GPU traffic goes over PCIe.
* **Default:** run independent single-GPU jobs, one per GPU (sweeps and seeds are embarrassingly parallel). Use DDP only when wall-clock for a single headline run matters.
* 16 GB limits the 124M micro-batch: use bf16 autocast with fp32 master weights, gradient accumulation, `torch.compile` and flash SDPA. Activation checkpointing is only needed if something larger does not fit.
* **Measure, don't assume, throughput** (Phase 0). Plan every budget from measured tokens/s.

### 7.2 Environment checklist
```bash
nvidia-smi                       # both GPUs visible, driver recent enough for Blackwell
python -c "import torch;print(torch.__version__, torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], torch.cuda.get_device_capability(0))"
# expect capability (12, 0). Needs a current stable PyTorch built for CUDA >= 12.8 (cu128 or newer
# wheels); older cu11x/cu121 builds do not support sm_120.
nproc; free -g; df -h            # CPU cores (data aug workers), RAM, disk (need ~100 GB free)
```
* **Python deps:** torch, torchvision, numpy, tiktoken, datasets, huggingface_hub, pandas, matplotlib, pyyaml, tensorboard, tqdm, (wandb), (timm). Baseline packages where available: `adam-mini`, `schedulefree`, `dog-optimizer`, `dadaptation`, `prodigyopt`, plus official Muon code.
* **Settings:** `torch.backends.cuda.matmul.allow_tf32=True`, `torch.backends.cudnn.allow_tf32=True`, `torch.set_float32_matmul_precision("high")`.
* Use tmux or nohup for long jobs. Every run must be **resumable** (§12).

### 7.3 Data
* **LM:** FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, `sample-10BT`).
  * Tokenize with tiktoken `gpt2` into uint16 `.bin` shards of 100M tokens; the first shard is validation.
  * About 5B tokens (~10 GB) is enough. A common pattern is build-nanogpt's `fineweb.py`.
* **Vision:** CIFAR-10/100 (torchvision). Tiny-ImageNet (public) is optional. ImageNet or ImageNet-100 only if the owner provides access.
* **Fixed evaluation sets.** LM: a fixed list of validation windows, e.g. 20M tokens for final eval and 2–5M for periodic eval. Vision: 5k images held out from train for tuning; the test set is reported once.

### 7.4 Repository layout (suggested)
```
hgsgd/
  CLAUDE.md  RESEARCH_BRIEF.md  RESEARCH_LOG.md  RESULTS.md  related_work.md
  src/hgsgd/
    optim/reference/   horizon_hd.py, anchored_sensitivity.py   (Appendix A, verbatim)
    optim/fast/        foreach / flat-buffer implementations (must match reference)
    optim/baselines/   hd.py, mechanic.py (per-block option), sgd_ll.py, grafting.py,
                       sgd_sai.py, wrappers for adam-mini / schedulefree / dog / prodigy / muon
    models/            gpt.py (nanoGPT-style), llama.py (RMSNorm/RoPE/SwiGLU), vit.py, resnet.py
    data/              fineweb.py, vision.py
    diagnostics/       hvp.py (autograd + finite diff), hessian.py (per-block power iteration,
                       Hutchinson trace), rayleigh.py, signals.py
    train_lm.py  train_vision.py  config.py  registry.py  logging.py
  configs/             yaml per experiment
  scripts/             queue.py (2-GPU job runner), prepare_fineweb.py, summarize.py, plots.py
  tests/               test_identities.py, test_reference.py, test_fast_vs_reference.py,
                       test_hvp.py, test_resume.py, test_ddp.py
  runs/                (gitignored) one dir per run: config.json, metrics.jsonl, final.json, ckpt/
  reports/             phase reports, figures
```

### 7.5 Two-GPU job runner
* A queue file (`jobs.jsonl`) and two workers, `CUDA_VISIBLE_DEVICES=0` and `=1`, each pulling the next pending job.
* Status is tracked in the registry (pending/running/done/failed/invalid). Workers retry once on failure and log the reason.
* Each run directory contains: config (with git SHA), `metrics.jsonl`, `final.json`, checkpoints and stdout.
* `scripts/summarize.py` regenerates `RESULTS.md` tables from the registry.

---

## 8. Experimental program (phases, gates, defaults)

**Model ladder (LM)** (measure throughput and fill in the budget table in §13):

| name | config | context | tokens/run | role |
|---|---|---|---|---|
| P1 | 6L, d = 384, 6 heads, untied embeddings | 512 | ~0.3B | fast sweeps |
| P2 | 8L, d = 512, 8 heads, untied | 1024 | ~1B | main ablations |
| M | GPT-2 small: 12L, d = 768, 12 heads, untied | 1024 | ~2.5B (≈ 20 tokens/param) | headline |
| M-L | LLaMA-style 124M (RMSNorm, RoPE, SwiGLU) | 1024 | ~2.5B | architecture generality (1–2 seeds) |
| (stretch) | ~350M | 1024 | 2–3B | scaling trend |

* Use a nanoGPT-style GPT-2: pre-LN, GELU, learned positional embeddings, no biases in linear layers, vocabulary padded to 50304.
* Untie the embeddings so the embedding and unembedding are separate blocks; this makes the per-row analysis clean.
* Default global batch is about 256K tokens via accumulation, with a batch-size ablation at {64K, 256K, 1M}.
* The hypergradient uses the **accumulated** optimizer-step gradient.

**Vision ladder:**
* ViT on CIFAR-100: patch 4, depth 6–8, width 384, 6 heads, ~200 epochs, RandAugment/AutoAugment, label smoothing 0.1, optional mixup/cutmix. Keep the data on GPU where possible.
* ResNet-18 on CIFAR-10/100 as the CNN control.
* Optional: ViT-S on Tiny-ImageNet, or ImageNet-100 if provided.
* Use a 5k validation split for tuning.

### Phase 0: setup and calibration (gate G0)
* Environment, data preparation, models, trainer, logging, registry and queue.
* Reproduce an AdamW 124M FineWeb-Edu curve and compare its shape and level against public build-nanogpt / modded-nanogpt references (note any differences in tokenizer, context or schedule).
* Measure tokens/s for P1, P2 and M, and images/s for the vision models.
* Fresh literature check (§4).
* **G0 deliverable:** throughput table, sanity curve, budget plan. Default: proceed.

### Phase 1: correctness (gate G1)
* Implement the tests in Appendix B.
* Port the reference optimizers (Appendix A) verbatim, write the fast implementation, and prove equivalence.
* **G1:** all tests green. **No research runs before G1.**

### Phase 2: feasibility ceiling, the "oracle ladder" (gate G2, the most important gate)
Run on P2 (2 seeds) and on ViT/CIFAR-100. All runs use the same tuning budget per method.

| exp | description |
|---|---|
| E2.1 | SGD-M (heavy-ball 0.9), one global LR, warmup + cosine, tuned |
| E2.2 | SGD-M with per-type LR multipliers (emb, head, LN/bias, attn, MLP) via a short coordinate search |
| E2.3 | **Grafting oracle per tensor:** SGD-M direction rescaled per tensor to AdamW's update norm, with AdamW run on the same batches (oracle only; uses Adam states) |
| E2.4 | E2.3 plus per-row grafting for the embedding and unembedding |
| E2.5 | Frozen-ratio oracle: time-averaged per-block ratios from E2.3/E2.4 used as fixed multipliers × a global schedule |
| E2.6 | AdamW, tuned |
| E2.7 | SGD-LL (re-implement from Glentis et al. 2026; read their paper for exact rules and LRs) |
| E2.8 | MDA (Jelassi & Defazio 2020; reference code in the facebookresearch/madgrad repository): anchored, prescribed global scale; also an "MDA per type" variant with tuned per-type constants |

* Metric: **gap closure** $=(L_{\rm SGDM}-L_{\rm X})/(L_{\rm SGDM}-L_{\rm AdamW})$.
* **G2 decision:**
  * If E2.3 ≥ 80% closure: per-tensor α suffices.
  * If only E2.4 does: rows are required for the embedding and unembedding.
  * If E2.5 ≈ E2.3: learning the scale matters more than learning the schedule over time.
  * If even E2.4 < 50%: the per-block scalar premise is weak for LMs. **Default branch:** move to Adam-mini-partition granularity (per head/neuron) and make ViT the primary testbed; tell the owner.

### Phase 3: method bring-up and mechanism diagnostics on P2 (gate G3)
* **E3.0** Bring up HorizonHD with the §2 configuration and a coarse sweep over α₀ × μ (pre-registered). Fix instabilities using the playbook (§11).
* **E3.1** Log per-block α trajectories and compare them with E2.3's grafted Adam magnitudes (correlation over time). At 10 checkpoints, compute per-block $\lambda^{(k)}_{\max}$ by power iteration with per-block HVPs on fixed batches, then plot $\alpha_k$ against $1/\lambda^{(k)}_{\max}$ and the product $\alpha_k\lambda^{(k)}_{\max}$ (H5).
* **E3.2** Retroactive-move diagnostics: per-block Rayleigh quotient $R_t$ along $D=\theta_t-\text{anchor}$ versus $\lambda_{\max}$; the norm ratio of retroactive move to SGD step; statistics of cos(g, S) (H4).
* **E3.3** (optional) Noise-floor check: compare mean $h$ with Lemma 7, using a per-block tr C estimated from the variance of microbatch gradients (free with gradient accumulation).
* **G3:** the method is stable, learns a non-trivial spread of per-block α, and beats tuned SGD-M on P2. **Default if not:** try playbook items 1–6 in §11, then report.

### Phase 4: ablations on P1/P2, plus key ones on ViT/CIFAR (gate G4)
Priority order, 2 seeds each, pre-registered:
* **A1** β ∈ {0, 0.9, 0.99, 0.999, 1} × r ∈ {0, 1}.
* **A2** normalization: cos, rms, natural, exp, raw.
* **A3** hypergradient: v1, v2 (cheap block-diagonal), v2 (exact per-block, one full HVP, Lemma 4), Level-K unrolled (K ∈ {2, 4}, P1 only), exact non-retro sensitivity (SMD; ρ ∈ {0.99, 1}), exact anchored $q$ recursion. Report per step **and** per wall-clock, plus the curvature-to-first-order ratio and sign-flip frequency (§3.6).
* **A4** granularity: global, type, tensor, tensor + rows(emb/head), Adam-mini partition, plus the hierarchical row variant.
* **A5** momentum: none, heavy ball 0.9, Nesterov, single-buffer HB-HD.
* **A6** external schedule: none, warmup only, warmup + cosine, WSD.
* **A7** robustness: α₀ × {1/30, 1/10, 1/3, 1, 3, 10, 30} and μ × {1/3, 1, 3}.
* **A8** S update: $S^+=S+\gamma\alpha^pg$ with p ∈ {0, 1} (and γ = 1/α as an equivalence check), compared at matched effective LR.
* **A9** clipping: none, global norm 1.0, SGD-LL layer-wise clipping combined with the method.
* **A10** batch size {64K, 256K, 1M tokens}.
* **A11** meta-step clip c ∈ {0.02, 0.05, 0.2} and bounds; positivity parameterization (log/exp vs softplus vs projection).
* **A12** outer objective: training-batch vs held-out-batch hypergradient (§3.16), on ViT/CIFAR-100 and P1 (H13).
* **G4:** pick 1–2 final variants plus a "minimal" variant (fewest knobs). Write the ablation report.

### Phase 5: headline results (gate G5)
* **LM (M, 124M, ~2.5B tokens, 3 seeds):** final variants versus Tier-A baselines (§9), plus Tier B where budget allows. Then M-L (LLaMA-style) for the final variant against AdamW and SGD-M (1–2 seeds).
* **Vision (3 seeds):** ViT/CIFAR-100 and ResNet-18/CIFAR-10/100 versus SGD-M, AdamW and the best baselines.
* **Report:** validation loss versus tokens and versus wall-clock, final metrics with 95% CIs, memory, step-time overhead, and α trajectories.

### Phase 6: generalization and sharpness (owner's core interest)
* On vision models, at matched training loss, compare test accuracy, generalization gap, $\lambda_{\max}$, Hessian trace (Hutchinson), per-block sharpness, and loss along random and filter-normalized directions.
* Optional SAM baseline. Test H10 and the Lookahead-style "retroactive pull" effect (r = 1 vs r = 0).

### Phase 7: scaling (stretch) and paper support
* A ~350M model for the best variant against AdamW and SGD-LL, if the budget allows.
* Turn §3 into lemma/proof form; produce figures and tables (§14).

---

## 9. Baselines and the fair tuning protocol

* **Tier A (mandatory):** SGD-M (heavy ball, tuned LR, warmup + cosine); AdamW (tuned); **SGD-LL**; **per-block HD** (Baydin, multiplicative cos form, β = 0 member, also global); **per-block Mechanic** on SGD-M (also global); grafting oracle (Phase 2 only).
* **Tier B:** **MDA** (anchored, prescribed scale; already matches Adam on some transformer tasks); Adam-mini; SGD-SaI; L-DoG; Muon (hidden matrices) + AdamW (the rest); Schedule-Free SGD.
* **Tier C:** MADGRAD; Prodigy / D-Adapt SGD with layer_scale; GALA; LANTON; MARTHE (global LR, forward-mode sensitivity).

**Tuning protocol (pre-register):**
* Each method gets the **same number of tuning runs** at proxy scale (e.g. 8). Learning rates use log grids with a factor of 2 or √10 around literature defaults.
* AdamW: β = (0.9, 0.95), ε = 1e-8, wd 0.1. SGD-M: momentum 0.9. Warmup is 2% of steps for all scheduled methods; decay is cosine to 10% of peak, plus a WSD check.
* Grad clipping is 1.0 for all methods unless a method's paper specifies otherwise (SGD-LL has its own clipping).
* For our method, the tuning budget covers α₀ × μ (and β if not fixed). Other knobs stay at defaults chosen on P1, and that P1 tuning is counted in the budget.
* Transfer the best settings to M with a 3-point local re-sweep for **every** method.
* Report best-of-grid **and** full sensitivity curves. Never report best-of-seeds.

---

## 10. Metrics, logging, statistics, reporting

* **Primary (LM):** validation loss at the end (fixed 20M-token eval set) and validation loss versus tokens and wall-clock.
* **Vision:** test top-1 (reported once), validation top-1 (for tuning), training loss.
* **Derived:** gap closure; tokens-to-target (tokens to reach AdamW's final loss); LR-sensitivity width (decades of α₀ within 2% of best); divergence rate; memory (peak allocated); optimizer step time ÷ total step time.
* **Optimizer internals** (every N steps, summarized by block type and depth, with the full vector dumped every 10N): α per block; cos(g, S) mean/std; ‖S‖, ‖g‖, ‖θ − anchor‖; the norm ratio of retroactive move to SGD step; fraction of blocks at bounds or clipped meta-steps; effective per-block update RMS.
* **Retroactive-share diagnostic** per block: $R_{k,t}=|\alpha_{k,t+1}-\alpha_{k,t}|\,\|S^{(k)}_t\|/(\alpha_{k,t+1}\|g^{(k)}_t\|+\varepsilon)$. Large values mean the method behaves mainly as displacement rescaling rather than as a modest LR adjustment.
* **Stability margins** (periodically): $\alpha_k\lambda^{(k)}_{\max}$; $\mu\,S^{(k)\top}HS^{(k)}$ (raw meta-step margin, should stay < 2); $\lambda^{(k)}_{\max}(\alpha+\mu\|S\|^2)$ (§3.4).
* **Diagnostics** (checkpoints): per-block $\lambda_{\max}$, Hutchinson trace, Rayleigh quotient along D.
* **Statistics:** 3 seeds for headline runs and 2 for ablations. Each seed fixes initialization and data order, and the same seeds are reused across methods. Report mean ± std and 95% CIs of **paired** differences.
* **Phase report template** (`reports/phase_N.md`): goal; pre-registered design; results table (with run IDs); figures; hypothesis verdicts; surprises and bugs; decision at the gate; next steps.

---

## 11. Iteration playbook and idea backlog

### 11.1 Symptom → likely cause → fix
| symptom | likely cause | try |
|---|---|---|
| early divergence | α grows too fast; spikes | lower μ or c; start meta-learning after K steps; tighter upper bound; global clipping; SGD-LL layer clipping |
| α collapses early, slow progress | short horizon or noise-floor drift | larger β; rms normalization; exact sensitivity; check that h uses the old S |
| α frozen, no annealing (β = 1) | dilution/amplification (P1/P2) | β < 1, r = 0, natural-gradient α step, external schedule |
| output-row NaNs or explosions | runaway rows (§3.7) | relative bounds, hierarchical shrinkage, skip rows with tiny gradient |
| matches SGD-M but not AdamW | granularity too coarse | rows for emb/head; Adam-mini partition; compare against the oracle at that granularity |
| good per step, poor per wall-clock | Python loops, HVP cost | foreach or flat buffers; amortized HVP; compile the optimizer step |
| large seed variance | noisy eval or unstable meta-dynamics | bigger eval set; smaller c; more seeds |

### 11.2 Idea backlog (ordered by expected value)
1. Hessian-block granularity with hierarchical shrinkage of row/head rates.
2. Multi-horizon ensemble: several β's, combined like Mechanic's sum or through online expert weights.
3. Exact sensitivity with finite-difference HVPs every m steps, or Gauss–Newton products.
4. A per-block coin-betting tuner replacing gradient steps on log α.
5. Separating scale from schedule: $\alpha_k(t)=s_k\,\eta(t)$, with $s_k$ learned on a long horizon and $\eta(t)$ global.
6. Combining with SGD-LL's layer-wise update clipping (stability) and learning the per-block scales on top.
7. Hypergradient-learned momentum per block (as in HDM-HB).
8. Warm-starting α from SGD-SaI's gradient SNR or a short Adam probe.
9. A noise-aware meta-signal: use microbatch gradient variance to separate the "schedule" component (noise drift, Lemma 7) from the "scale" component.
10. A trust region on the retroactive move: $\|\Delta\alpha\,S\|\le\kappa\,\|\alpha g\|$.
11. Held-out-loss hypergradient (§3.16), possibly only for the scale component, with the schedule still from the training signal.
12. Learned per-block scales on top of MDA's prescribed schedule ($\alpha_k(t)=s_k/\sqrt{t+1}$ with $s_k$ learned): isolates scale learning from schedule learning inside the dual-averaging family.
13. **Pivot:** apply HorizonHD as a per-block LR wrapper on AdamW or Muon (broader impact).

### 11.3 Pivot criteria (ask the owner)
* If after G4 the best variant does not beat tuned SGD-M by a clear margin on P2 (> 3× the seed std) **and** does not reach ≥ 50% gap closure, propose to the owner one of:
  * (a) the wrapper pivot (backlog 11);
  * (b) an analysis paper (C1 + C3 + C4);
  * (c) a vision-first focus.

---

## 12. Engineering pitfalls

* **Precision.** Keep S, q, log α and momentum in fp32; reduce dot products in fp32; never accumulate S in bf16.
* **β = 1 drift.** The incremental update $\theta\mathrel{-}=\alpha_{t+1}g+\Delta\alpha S$ drifts from $\theta_0-\alpha S$ in floating point. Either recompute from a stored $\theta_0$ (one more buffer) or periodically re-sync and log the drift.
* **Order of operations.** h uses the *old* S; the meta-step comes before the parameter step; the parameter step uses $\alpha_{t+1}$; S is updated last.
* **First step.** S = 0, so the cosine is undefined and the true hypergradient is 0. The ε guard gives Δ = 0; skip the meta-update at t = 0.
* **Ordering bias.** Computing h after adding $g_t$ to S adds $+\|g_t\|^2$ (§3.2). Test for it.
* **Scalar identity for v2.** If the previous gradient's graph is retained but the current α is a fresh leaf, the Hessian term silently vanishes (§3.6). Test with finite differences (Appendix B, T2).
* **Effective-weight cost.** A functional `anchor - alpha*S` implementation materializes weights in every forward pass; a fused optimizer that keeps θ and updates it incrementally is cheaper (with the drift check above).
* **Clipping.** Compute h from the same (clipped) gradient that is used for the update, and log the clipping rate.
* **Weight decay.** Use decoupled decay, kept out of S and h. Because α spans orders of magnitude, "lr × wd" coupling makes decay wildly different across blocks; start with wd = 0, then test a fixed per-step decay that does not depend on α.
* **Gradient accumulation and DDP.** Use the all-reduced, accumulated gradient. All ranks compute identical α; assert equality periodically or broadcast from rank 0.
* **Performance.** About 150 tensors per step; Python per-tensor loops add milliseconds. Use `torch._foreach_*` ops or flat buffers with segment reductions (`index_add_`) for per-block dot products. Target optimizer overhead < 5% of step time.
* **HVP with SDPA.** Double backward fails with flash kernels (§3.6).
* **Resumability.** Checkpoint model, all optimizer state (S, q, θ₀ if used, log α, rms state, momentum, step), RNG states and the data-loader position. Test that resume equals an uninterrupted run.
* **Logging cost.** Avoid `.item()` syncs every step.
* **Fairness traps.** Same token budget, same eval windows, same data order, same warmup for scheduled baselines. Do not compare against untuned or stale baseline numbers from papers; reproduce baselines in our pipeline.

---

## 13. Compute budget (fill in after Phase 0)

$$\text{GPU-hours}=\frac{\text{tokens per run}}{\text{measured tokens/s}\times3600}\times\#\text{runs}$$
Two GPUs give 48 GPU-hours per wall-clock day.

| phase | runs (approx.) | scale | GPU-h (fill in) |
|---|---|---|---|
| 0–1 | sanity + tests | P1/M short | |
| 2 | 7 methods × ~6 tunings × 2 seeds (+ vision) | P2, ViT | |
| 3 | ~20 | P2 | |
| 4 | ~80–120 | P1/P2 (+ vision subset) | |
| 5 | ~6 methods × 3 seeds + local re-sweeps (+ vision) | M, ViT, ResNet | |
| 6–7 | diagnostics, stretch | vision, 350M | |

Keep a running `compute_ledger.csv`. If the projected total exceeds about 4 weeks of wall-clock, shrink Phase 4 first (move ablations to P1) and keep Phase 5 intact.

---

## 14. Publication plan

* **Framing A (method paper; needs a strong C2):** "SGD with learned per-block learning rates closes the transformer gap." Requirements:
  * GPT-2 124M (and the LLaMA-style variant) within ~2% of AdamW, or better than SGD-LL, with SGD-family memory and no hand-designed clipping.
  * ViT/CIFAR at least on par with AdamW.
  * No regression on ResNet.
  * LR robustness (H7).
* **Framing B (analysis + method):** the horizon family (C1) with the mechanisms (noise-floor drift, ray dilution/amplification, per-block edge of stability) validated at scale (C3), plus a solid method result.
* **Framing C (theory/analysis, workshop or TMLR):** if C2 is weak.
* **Must-have figures and tables:**
  1. Validation loss vs tokens and vs wall-clock for all methods (124M).
  2. Gap-closure bar chart with oracles.
  3. Learned α per block over time (heatmap: layer × block type).
  4. α_k vs 1/λ_max^{(k)}, and the per-block edge-of-stability product.
  5. Horizon ablation (β × r).
  6. LR-sensitivity curves.
  7. Granularity ablation.
  8. Memory and overhead table.
  9. Vision accuracy and sharpness table.
* **Theory tasks:**
  * Write lemmas 1–8 with proofs (most are in §3).
  * Lemma 7 as a theorem for the noisy quadratic model (per-direction weights).
  * Dilution/amplification bounds.
  * Optional: stochastic regret of bounded-log-step horizon-HD against the best fixed per-block α in hindsight (extending Chu et al. 2025 beyond the deterministic setting).
* **Venues:** ICML / NeurIPS 2027, TMLR, or the NeurIPS OPT workshop. Check the current deadlines.

---

## 15. References

* Almeida et al. (1998), *Parameter adaptation in stochastic optimization*.
* Baydin et al. (2018), *Online Learning Rate Adaptation with Hypergradient Descent*. arXiv:1703.04782
* Chu, Gao, Ye, Udell (ICML 2025), *Provable and Practical Online Learning Rate Adaptation with Hypergradient Descent*. arXiv:2502.11229
* Cutkosky, Defazio, Mehta (NeurIPS 2023), *Mechanic: A Learning Rate Tuner*. arXiv:2306.00144 (optax `contrib.mechanize`)
* Orabona & Pál (NeurIPS 2016), *Coin Betting and Parameter-Free Online Learning*. arXiv:1602.04128
* Defazio & Mishchenko (2023), *D-Adaptation*. arXiv:2301.07733 · Mishchenko & Defazio (ICML 2024), *Prodigy*. arXiv:2306.06101
* Ivgi, Hinder, Carmon (ICML 2023), *DoG is SGD's Best Friend* (L-DoG). arXiv:2302.12022
* Schraudolph (1999), *Local gain adaptation in stochastic gradient descent* (SMD) · Sutton (1992), IDBD
* Wu, Ren, Liao, Grosse (ICLR 2018), *Understanding Short-Horizon Bias in Stochastic Meta-Optimization*. arXiv:1803.02021
* McCandlish et al. (2018), *An Empirical Model of Large-Batch Training*. arXiv:1812.06162
* Zhang et al. (NeurIPS 2024), *Why Transformers Need Adam: A Hessian Perspective*. arXiv:2402.16788
* Tomihari & Sato (TMLR 2026), *Understanding Transformer Optimization via Gradient Heterogeneity*. arXiv:2502.00213
* Kunstner et al. (NeurIPS 2024), *Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models*. arXiv:2402.19449
* Zhao et al. (ICLR 2025), *Deconstructing What Makes a Good Optimizer for (Autoregressive) Language Models*. arXiv:2407.07972
* Glentis, Li, Yau, Hong (2026), *Revisiting the Adam-SGD Gap in LLM Pre-Training: The Role of Large Effective Learning Rates* (SGD-LL). arXiv:2605.17787
* Zhang et al. (ICLR 2025), *Adam-mini: Use Fewer Learning Rates To Gain More*. arXiv:2406.16793
* Xu et al. (2024), *No More Adam: Learning Rate Scaling at Initialization is All You Need* (SGD-SaI). arXiv:2412.11768
* Marek et al. (NeurIPS 2025), *Small Batch Size Training for Language Models: When Vanilla SGD Works…*. arXiv:2507.07101
* Henheik, Eimer, Lindauer (2025), *Revisiting Learning Rate Control*. arXiv:2507.01724
* Jiang, Kavis, Mokhtari (2025), *Online Learning-guided Learning Rate Adaptation via Gradient Alignment* (GALA). arXiv:2506.08419
* Hao et al. (2025), *Noise-Adaptive Layerwise Learning Rates* (LANTON). arXiv:2510.14009
* He et al. (2026), *One LR Doesn't Fit All: Heavy-Tail Guided Layerwise Learning Rates for LLMs*. arXiv:2605.22297
* Li et al. (ACL 2025), *Taming LLMs by Scaling Learning Rates with Gradient Grouping* (SGG). arXiv:2506.01049
* Zhang, Li, Nado, Martens, Sachdeva, Dahl, Shallue, Grosse (NeurIPS 2019), *Which Algorithmic Choices Matter at Which Batch Sizes? Insights from a Noisy Quadratic Model*. arXiv:1907.04164
* Qi et al. (ICLR 2026), *DNT: A Deeply Normalized Transformer That Can Be Trained by Momentum SGD*.
* Agarwal et al. (2020), *Disentangling Adaptive Gradient Methods from Learning Rates* (grafting). arXiv:2002.11803
* Defazio et al. (NeurIPS 2024), *The Road Less Scheduled* (Schedule-Free). arXiv:2405.15682
* Jordan et al. (2024), Muon (blog) · Liu et al. (2025), *Muon is Scalable for LLM Training*. arXiv:2502.16982
* Cohen et al. (ICLR 2021), *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability*. arXiv:2103.00065
* Lucas et al. (ICML 2021), *Analyzing Monotonic Linear Interpolation in Neural Network Loss Landscapes*. arXiv:2104.11044
* Zhang, Lucas, Hinton, Ba (NeurIPS 2019), *Lookahead Optimizer: k steps forward, 1 step back*. arXiv:1907.08610
* Nesterov (2009), *Primal-dual subgradient methods for convex problems* · Xiao (JMLR 2010), *Dual Averaging Methods for Regularized Stochastic Learning and Online Optimization*
* Jelassi & Defazio (2020), *Dual Averaging is Surprisingly Effective for Deep Learning Optimization* (MDA). arXiv:2010.10502
* Defazio & Jelassi (JMLR 2022), *A Momentumized, Adaptive, Dual Averaged Gradient Method* (MADGRAD). arXiv:2101.11075
* Donini et al. (IJCAI 2020), *MARTHE: Scheduling the Learning Rate via Online Hypergradients*. arXiv:1910.08525
* Du, Hu, Lee (NeurIPS 2018), *Algorithmic Regularization in Learning Deep Homogeneous Models: Layers are Automatically Balanced*. arXiv:1806.00900

---

## Appendix A: Reference implementations (verified; port verbatim to `src/hgsgd/optim/reference/`)

The code below was verified in float64 against finite differences, exact trajectory equivalences and an independent NumPy simulation (see Appendix B). It is **correct, not fast**: it uses Python loops over tensors and autograd double-backward HVPs. Build fast versions (foreach or flat buffers, finite-difference HVP) and prove equivalence to this code before using them.

* `HorizonHD`: the β/r family with first-order signal, per tensor, per row or per group; optional momentum.
* `SensitivityHD`: exact non-retroactive sensitivity (scalar-gain SMD per block), one HVP per step.
* `AnchoredSensitivityHD`: anchored θ₀ − αS with the owner's one-step HVP correction (`mode="v2"`) or the full-horizon exact recursion (`mode="full"`). Per-block exactness caveat: Lemma 4.

### A.1 `horizon_hd.py`
```python
"""
Per-block hypergradient learning rates for SGD with a tunable hypergradient horizon.

For every block k (a parameter tensor, a row of a matrix, or a whole param group):

    d_t      = g_t                     (or the heavy-ball buffer m_t = momentum*m_{t-1} + g_t)
    h_t      = < g_t , S_t >           (S_t: beta-discounted sum of past directions;
                                        h_t = -dL_t/d alpha for theta_t = a_t - alpha * S_t)
    log a   += clip(meta_lr * normalize(h_t))          ('cos' or 'rms' normalization)
    theta   -= a_{t+1} * d_t  +  retro * beta * (a_{t+1} - a_t) * S_t
    S_{t+1}  = beta * S_t + d_t

    beta = 0            -> hypergradient descent (Baydin et al. 2018, multiplicative form)
    beta = 1, retro = 1 -> the anchored reparameterization theta_t = theta_0 - alpha_t * S_t
                           (same parameterization as Mechanic / KT coin betting, GD tuner)
    0 < beta < 1        -> anchor = EMA of past iterates with horizon ~ 1/(1 - beta)

The retroactive term rescales the recent path when alpha changes; with retro = 1 the iterate
stays exactly on  theta_t = anchor_t - alpha_t * S_t  (anchor = EMA of iterates, decay beta).
Memory: one buffer (S) per parameter, plus the momentum buffer if momentum > 0.
Reference code: assumes fp32 (or fp64) parameters (fp32 master weights under bf16 autocast); S is
allocated in the parameter dtype. Python loops over tensors: correct but slow; see brief section 12.

Caution: the cosine-normalized meta-step is scale-free, so a block whose
gradient keeps the same direction while the loss has no minimum along it (e.g. output rows of
classes that never occur) grows its rate exponentially and eventually blows up. With block="row"
set an upper bound relative to lr, e.g. lr_bounds=(lr * 1e-4, lr * 100).

Usage:
    opt = HorizonHD(model.parameters(), lr=1e-2, meta_lr=1e-2, beta=0.99)
    # rows of embedding / unembedding matrices as separate blocks (token-frequency heterogeneity):
    opt = HorizonHD([{"params": body_params},
                     {"params": [emb.weight, head.weight], "block": "row"}], lr=1e-2)
    ...
    loss.backward(); opt.step(); opt.zero_grad()
    opt.block_lrs()   # dict name/index -> current per-block learning rates (for logging)
"""
import math
import torch
from torch.optim import Optimizer


class HorizonHD(Optimizer):
    def __init__(self, params, lr=1e-2, meta_lr=1e-2, beta=0.99, retro=1.0, norm="cos",
                 momentum=0.0, weight_decay=0.0, lr_bounds=(1e-8, 1e3), max_log_step=0.05,
                 rms_beta=0.99, block="tensor", eps=1e-30):
        if norm not in ("cos", "rms"):
            raise ValueError("norm must be 'cos' or 'rms'")
        if block not in ("tensor", "row", "group"):
            raise ValueError("block must be 'tensor', 'row' or 'group'")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        defaults = dict(lr=lr, meta_lr=meta_lr, beta=beta, retro=retro, norm=norm,
                        momentum=momentum, weight_decay=weight_decay, lr_bounds=lr_bounds,
                        max_log_step=max_log_step, rms_beta=rms_beta, block=block, eps=eps)
        super().__init__(params, defaults)

    # ----------------------------------------------------------- block helpers
    @staticmethod
    def _rowwise(p, block):
        return block == "row" and p.dim() >= 2

    @staticmethod
    def _bdot(a, b, rowwise):
        if rowwise:
            return (a * b).flatten(1).sum(1)          # (rows,)
        return (a * b).sum()

    @staticmethod
    def _expand(x, p, rowwise):
        return x.view(-1, *([1] * (p.dim() - 1))) if rowwise else x

    @staticmethod
    def _cdtype(p):
        """dtype for the meta computations: float64 params stay float64, everything else float32"""
        return torch.float64 if p.dtype == torch.float64 else torch.float32

    def _init_state(self, p, group):
        st = self.state[p]
        rowwise = self._rowwise(p, group["block"])
        shape = (p.shape[0],) if rowwise else ()
        cd = self._cdtype(p)
        st["S"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        st["log_lr"] = torch.full(shape, math.log(group["lr"]), dtype=cd, device=p.device)
        st["vh"] = torch.zeros(shape, dtype=cd, device=p.device)
        st["step"] = 0
        if group["momentum"] > 0:
            st["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        return st

    # ---------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            beta, mom, norm = group["beta"], group["momentum"], group["norm"]
            lo, hi = (math.log(b) for b in group["lr_bounds"])
            states = [self.state[p] if len(self.state[p]) else self._init_state(p, group) for p in params]

            # 1) per-block hypergradient signal  h = <g, S>
            if group["block"] == "group":
                cd = self._cdtype(params[0])
                h = sum(self._bdot(p.grad.to(cd), st["S"].to(cd), False) for p, st in zip(params, states))
                gg = sum(self._bdot(p.grad.to(cd), p.grad.to(cd), False) for p in params)
                ss = sum(self._bdot(st["S"].to(cd), st["S"].to(cd), False) for st in states)
                gstate = states[0]                    # group-level alpha lives in the first param's state
                new_log = self._meta_update(gstate, h, gg, ss, group, lo, hi)
                for st in states[1:]:
                    st["log_lr"] = new_log            # keep a shared copy for logging
                for p, st in zip(params, states):
                    self._param_update(p, st, group, gstate["_old_lr"], new_log.exp(), False)
            else:
                for p, st in zip(params, states):
                    rowwise = self._rowwise(p, group["block"])
                    cd = self._cdtype(p)
                    g = p.grad.to(cd)
                    S = st["S"].to(cd)
                    h = self._bdot(g, S, rowwise)
                    gg = self._bdot(g, g, rowwise)
                    ss = self._bdot(S, S, rowwise)
                    new_log = self._meta_update(st, h, gg, ss, group, lo, hi)
                    self._param_update(p, st, group, st["_old_lr"], new_log.exp(), rowwise)
        return loss

    def _meta_update(self, st, h, gg, ss, group, lo, hi):
        st["step"] += 1
        log_lr = st["log_lr"]
        st["_old_lr"] = log_lr.exp()
        eps = group["eps"]
        if group["norm"] == "cos":
            hn = h / (gg.sqrt() * ss.sqrt() + eps)       # product of norms: no under/overflow of gg*ss
        else:                                          # 'rms': Adam-normalized d L / d log(alpha)
            ah = st["_old_lr"] * h
            rb = group["rms_beta"]
            st["vh"].mul_(rb).add_((1 - rb) * ah * ah)
            vhat = st["vh"] / (1 - rb ** st["step"])
            hn = torch.where(vhat > 0, ah / (vhat.sqrt() + eps), torch.zeros_like(ah))
        m = group["max_log_step"]
        dlog = (group["meta_lr"] * hn).clamp(-m, m)
        st["log_lr"] = (log_lr + dlog).clamp(lo, hi)
        return st["log_lr"]

    def _param_update(self, p, st, group, old_lr, new_lr, rowwise):
        g = p.grad
        S = st["S"]
        beta, mom = group["beta"], group["momentum"]
        d = g
        if mom > 0:
            st["m"].mul_(mom).add_(g)
            d = st["m"]
        a_new = self._expand(new_lr, p, rowwise).to(p.dtype)
        if group["weight_decay"] > 0:                 # decoupled, kept out of S and the hypergradient
            p.mul_(1 - a_new * group["weight_decay"])
        if group["retro"] != 0 and beta > 0:
            da = self._expand(new_lr - old_lr, p, rowwise).to(p.dtype)
            p.sub_(group["retro"] * beta * da * S)     # retroactive rescaling of the (recent) path
        p.sub_(a_new * d)
        S.mul_(beta).add_(d)

    # ------------------------------------------------------------- logging
    @torch.no_grad()
    def block_lrs(self, names=None):
        """current learning rates; names: optional dict param -> name"""
        out = {}
        for gi, group in enumerate(self.param_groups):
            for pi, p in enumerate(group["params"]):
                st = self.state.get(p, {})
                if "log_lr" not in st:
                    continue
                key = names.get(p, f"g{gi}.p{pi}") if names else f"g{gi}.p{pi}"
                out[key] = st["log_lr"].exp().cpu()
        return out


class SensitivityHD(Optimizer):
    """
    Per-block learning rates driven by the exact forward-mode hypergradient (one HVP per step),
    i.e. scalar-gain stochastic meta-descent (Schraudolph 1999) at block granularity:

        h_t      = < g_t , -v_t >                     (-dL_t/d alpha, v ~ d theta_t / d alpha)
        log a   += clip(meta_lr * h_t / (|g_t| |v_t|))
        v_{t+1}  = rho * (v_t - a_{t+1} * H_t v_t) - g_t
        theta   -= a_{t+1} * g_t

    rho = 0 recovers HD; dropping the H v term with rho = 1 gives the anchored signal without
    the retroactive move. The H v term gives every curvature direction its own memory
    ~ 1/(alpha * lambda), which a single beta cannot.

    The closure must return the loss WITHOUT calling backward; the optimizer differentiates it:
        opt.step(lambda: loss_fn(model(x), y))
    """

    def __init__(self, params, lr=1e-2, meta_lr=3e-2, rho=0.999, lr_bounds=(1e-8, 1e3),
                 max_log_step=0.05, block="tensor", eps=1e-30):
        defaults = dict(lr=lr, meta_lr=meta_lr, rho=rho, lr_bounds=lr_bounds,
                        max_log_step=max_log_step, block=block, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure):
        params = [p for g in self.param_groups for p in g["params"] if p.requires_grad]
        with torch.enable_grad():
            loss = closure()
            grads = torch.autograd.grad(loss, params, create_graph=True)
        for p in params:
            st = self.state[p]
            if not st:
                grp = next(g for g in self.param_groups if any(q is p for q in g["params"]))
                rowwise = HorizonHD._rowwise(p, grp["block"])
                cd = HorizonHD._cdtype(p)
                st["v"] = torch.zeros_like(p)
                st["log_lr"] = torch.full((p.shape[0],) if rowwise else (), math.log(grp["lr"]),
                                          dtype=cd, device=p.device)
        vs = [self.state[p]["v"] for p in params]
        hv = torch.autograd.grad(grads, params, grad_outputs=vs)        # H_t v_t (one HVP)
        it = iter(zip(params, grads, hv))
        with torch.no_grad():
            for group in self.param_groups:
                lo, hi = (math.log(b) for b in group["lr_bounds"])
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    _, g, Hv = next(it)
                    g = g.detach()
                    st = self.state[p]
                    rowwise = HorizonHD._rowwise(p, group["block"])
                    cd = HorizonHD._cdtype(p)
                    v = st["v"]
                    gc, vc = g.to(cd), v.to(cd)
                    h = -HorizonHD._bdot(gc, vc, rowwise)
                    nrm = HorizonHD._bdot(gc, gc, rowwise).sqrt() * HorizonHD._bdot(vc, vc, rowwise).sqrt()
                    m = group["max_log_step"]
                    dlog = (group["meta_lr"] * h / (nrm + group["eps"])).clamp(-m, m)
                    st["log_lr"] = (st["log_lr"] + dlog).clamp(lo, hi)
                    a = HorizonHD._expand(st["log_lr"].exp(), p, rowwise).to(p.dtype)
                    v.mul_(group["rho"]).sub_(group["rho"] * a * Hv.detach()).sub_(g)
                    p.sub_(a * g)
        return loss.detach()

    @torch.no_grad()
    def block_lrs(self, names=None):
        return HorizonHD.block_lrs(self, names)
```

### A.2 `anchored_sensitivity.py`
```python
"""
Anchored reparameterization with curvature-corrected hypergradients (reference implementation).

Per block k:  theta_t = theta_0 - alpha_t * S_t,   S_{t+1} = S_t + g_t,   g_t = grad L_t(theta_t)
Because S_t depends on alpha through all past gradients, define q_t := dS_t/dalpha. Then
    d theta_t / d alpha = -(S_t + alpha_t q_t)
    h_t := -dL_t/dalpha = < g_t , S_t + alpha_t q_t >
mode = "v1"   : q = 0                                   (first-order: S_t detached; no HVP)
mode = "v2"   : q_{t+1} = -H_t S_t                      (one-step: the user's formula
                                                         h_t = <g_t, S_t - alpha H_{t-1} S_{t-1}>)
mode = "full" : q_{t+1} = rho * (q_t - H_t (S_t + alpha_t q_t))   (exact forward-mode, rho = 1;
                                                         holds alpha fixed over the past, RTRL-style)
H_t is the Hessian of the SAME minibatch loss L_t at theta_t (one HVP per step for v2/full).
alpha update: log alpha += clip(meta_lr * h / (|g| |S + alpha q|)); then theta = theta_0 - alpha S (exact).

Reference code: float64-capable, autograd double-backward HVP (needs a twice-differentiable model;
flash/memory-efficient SDPA kernels are NOT; use the math SDPA backend or a finite-difference HVP).
The closure must return the loss WITHOUT calling backward.
"""
import math
import torch
from torch.optim import Optimizer


class AnchoredSensitivityHD(Optimizer):
    def __init__(self, params, lr=1e-2, meta_lr=3e-2, mode="v2", rho=1.0, lr_bounds=(1e-8, 1e3),
                 max_log_step=0.05, eps=1e-30):
        if mode not in ("v1", "v2", "full"):
            raise ValueError(mode)
        super().__init__(params, dict(lr=lr, meta_lr=meta_lr, mode=mode, rho=rho, lr_bounds=lr_bounds,
                                      max_log_step=max_log_step, eps=eps))
        self.last_h = []                                   # per-block h_t of the last step (for tests/logging)

    def _params(self):
        return [p for g in self.param_groups for p in g["params"] if p.requires_grad]

    def step(self, closure):
        params = self._params()
        for grp in self.param_groups:
            for p in grp["params"]:
                st = self.state[p]
                if not st:
                    st["theta0"] = p.detach().clone()
                    st["S"] = torch.zeros_like(p)
                    st["q"] = torch.zeros_like(p)
                    st["log_lr"] = torch.tensor(math.log(grp["lr"]), dtype=p.dtype, device=p.device)
        need_hvp = any(g["mode"] != "v1" for g in self.param_groups)
        with torch.enable_grad():
            loss = closure()
            grads = torch.autograd.grad(loss, params, create_graph=need_hvp)
        # vectors to multiply by H_t
        ws = []
        for grp in self.param_groups:
            for p in grp["params"]:
                st = self.state[p]
                a = st["log_lr"].exp()
                ws.append(st["S"] + a * st["q"] if grp["mode"] == "full" else st["S"])
        hvps = torch.autograd.grad(grads, params, grad_outputs=ws) if need_hvp else [None] * len(params)
        self.last_h = []
        i = 0
        with torch.no_grad():
            for grp in self.param_groups:
                lo, hi = (math.log(b) for b in grp["lr_bounds"])
                for p in grp["params"]:
                    g, Hw = grads[i].detach(), hvps[i]
                    i += 1
                    st = self.state[p]
                    a_old = st["log_lr"].exp()
                    u = st["S"] + a_old * st["q"]           # -(d theta_t / d alpha)
                    h = (g * u).sum()
                    self.last_h.append(h.clone())
                    # sensitivity update (uses H_t of this batch at theta_t)
                    if grp["mode"] == "v2":
                        st["q"] = -Hw.detach()
                    elif grp["mode"] == "full":
                        st["q"] = grp["rho"] * (st["q"] - Hw.detach())
                    # meta step on log alpha (cosine-normalized, clipped)
                    nrm = (g * g).sum().sqrt() * (u * u).sum().sqrt()
                    m = grp["max_log_step"]
                    dlog = (grp["meta_lr"] * h / (nrm + grp["eps"])).clamp(-m, m)
                    st["log_lr"] = (st["log_lr"] + dlog).clamp(lo, hi)
                    # accumulate and move along the reparameterization (exact, from theta_0)
                    st["S"].add_(g)
                    p.copy_(st["theta0"] - st["log_lr"].exp() * st["S"])
        return loss.detach()
```

---

## Appendix B: Tests the agent must have (and pass) before research runs

### B.1 Identity checks for §3 (`tests/test_identities.py`; NumPy; port this file)
Each line prints a relative error (≈1e-10 to 1e-16 for exact identities, Monte Carlo agreement for [7] and [8]).
* [1] Lemma 1 (closed form)
* [2] Lemma 2 (θ-space rank-one form)
* [3] Lemma 3 (β = 0 ≡ HD)
* [4] Lemma 8 (KT ≡ anchored)
* [5] the owner's v2 formula against finite differences
* [6] exact forward sensitivity against finite differences
* [7] Lemma 6 (same-batch vs fresh-batch optimal step)
* [8] Lemma 7 (noise-floor drift)
* [9] Lemma 5 (conservation law)

```python
"""
Numerical checks of the identities used in the write-up.
Each check prints the max relative error (should be ~1e-12 for exact identities).
"""
import numpy as np

rng = np.random.default_rng(0)
d = 30
lam = np.logspace(0, -3, d)                 # diagonal Hessian
B = 20.0
C = lam / B                                 # NQM noise covariance (diagonal)


def grad(th, z):
    return lam * th + np.sqrt(C) * z


def rel(a, b):
    return np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-300)


T = 200
Z = rng.standard_normal((T + 1, d))
th0 = rng.standard_normal(d) * 3

# ---------------------------------------------------------------- 1 & 2
# proposal: alpha_{t+1} = alpha_t + mu <g_t,S_t>, S_{t+1} = S_t + g_t, theta = theta0 - alpha S
mu, a = 1e-4, 0.05
S = np.zeros(d)
sumsq = 0.0
err_closed, err_theta = 0.0, 0.0
th = th0.copy()
for t in range(T):
    g = grad(th, Z[t])
    a_new = a + mu * g @ S
    S_new = S + g
    th_new = th0 - a_new * S_new
    sumsq += g @ g
    closed = 0.05 + 0.5 * mu * (S_new @ S_new - sumsq)                 # closed form
    err_closed = max(err_closed, abs(a_new - closed) / abs(closed))
    step_pred = -a_new * g - mu * (g @ S) * S                          # theta-space form
    err_theta = max(err_theta, rel(th_new - th, step_pred))
    a, S, th = a_new, S_new, th_new
print(f"[1] closed form alpha = alpha0 + mu/2 (|S|^2 - sum|g|^2):   rel err {err_closed:.1e}")
print(f"[2] theta step = -alpha_(t+1) g - mu <g,S> S (rank-one):     rel err {err_theta:.1e}")

# ---------------------------------------------------------------- 3
# leaky anchor with beta = 0 reproduces Baydin et al.'s HD exactly
mu, a0 = 1e-3, 0.05
# (a) HD as published: alpha_t = alpha_{t-1} + mu g_{t-1}.g_{t-2};  theta_t = theta_{t-1} - alpha_t g_{t-1}
th, a, gprev = th0.copy(), a0, np.zeros(d)
traj_hd = []
for t in range(T):
    g = grad(th, Z[t])
    a = a + mu * g @ gprev
    th = th - a * g
    gprev = g
    traj_hd.append(th.copy())
# (b) leaky anchor, beta = 0: theta = anchor - alpha S, anchor <- beta anchor + (1-beta) theta, S <- beta S + g
beta = 0.0
th, a, S, anc = th0.copy(), a0, np.zeros(d), th0.copy()
traj_la = []
for t in range(T):
    g = grad(th, Z[t])
    a = a + mu * g @ S
    anc = beta * anc + (1 - beta) * th
    S = beta * S + g
    th = anc - a * S
    traj_la.append(th.copy())
print(f"[3] leaky anchor (beta=0) == HD trajectory:                 rel err {rel(np.array(traj_la), np.array(traj_hd)):.1e}")

# ---------------------------------------------------------------- 4
# KT coin betting (Orabona & Pal 2016, Hilbert space) == anchored form with alpha_t = Wealth_{t-1}/t
gscale = 1.0 / (np.linalg.norm(lam * th0) + 5 * np.linalg.norm(np.sqrt(C)) + 1)   # keep |g| <= 1
eps0 = 1.0
# (a) as published, in terms of reward vectors c_t = -g_t and prediction w_t (anchor at th0)
W, csum = eps0, np.zeros(d)
traj_kt = []
for t in range(1, T + 1):
    w = (W / t) * csum                       # w_t = (1/t)(eps + sum <c_i,w_i>) sum c_i
    x = th0 + w
    g = gscale * grad(x, Z[t - 1])
    c = -g
    W = W + c @ w
    csum = csum + c
    traj_kt.append(x.copy())
# (b) anchored recursion: alpha_{t+1} = (t alpha_t + alpha_t <g_t,S_t>)/(t+1)
S, alpha = np.zeros(d), eps0 / 1.0
traj_an = []
for t in range(1, T + 1):
    x = th0 - alpha * S
    g = gscale * grad(x, Z[t - 1])
    alpha = (t * alpha + alpha * (g @ S)) / (t + 1)
    S = S + g
    traj_an.append(x.copy())
print(f"[4] KT coin betting == anchored form, alpha_t = Wealth/t:   rel err {rel(np.array(traj_an), np.array(traj_kt)):.1e}")

# ---------------------------------------------------------------- 5
# variant 2: d/dalpha L_t(theta0 - alpha (S_{t-1} + g_{t-1}(theta0 - alpha S_{t-1})))
#          = < g_t, -S_t + alpha H S_{t-1} >   (S_{t-1} detached)
Sm1 = rng.standard_normal(d)
z1, z2 = Z[5], Z[6]


def F(al):
    th_prev = th0 - al * Sm1
    St = Sm1 + grad(th_prev, z1)
    th_t = th0 - al * St
    return 0.5 * lam @ th_t ** 2 + (np.sqrt(C) * z2) @ th_t        # L_t with its own noise


al = 0.3
th_prev = th0 - al * Sm1
St = Sm1 + grad(th_prev, z1)
gt = grad(th0 - al * St, z2)
analytic = gt @ (-St + al * lam * Sm1)
h = 1e-6
fd = (F(al + h) - F(al - h)) / (2 * h)
print(f"[5] variant-2 hypergradient vs finite difference:           rel err {abs(analytic - fd) / abs(fd):.1e}")

# ---------------------------------------------------------------- 6
# exact forward sensitivity v_{t+1} = (I - alpha H) v_t - g_t  vs finite difference of unrolled loss
def unrolled(al, n=60):
    th = th0.copy()
    for t in range(n):
        th = th - al * grad(th, Z[t])
    return 0.5 * lam @ th ** 2 + (np.sqrt(C) * Z[n]) @ th, th


al, n = 0.4, 60
th, v = th0.copy(), np.zeros(d)
for t in range(n):
    g = grad(th, Z[t])
    v = v - al * lam * v - g
    th = th - al * g
gT = grad(th, Z[n])
analytic = gT @ v
fd = (unrolled(al + 1e-6, n)[0] - unrolled(al - 1e-6, n)[0]) / 2e-6
print(f"[6] exact forward sensitivity vs finite difference:          rel err {abs(analytic - fd) / abs(fd):.1e}")

# ---------------------------------------------------------------- 7
# greedy step size from same-batch vs fresh-batch hypergradients (HD's 'lookahead')
th = rng.standard_normal(d) * 0.05            # near the noise floor: |grad f|^2 comparable to tr C
gf = lam * th
n_mc = 400_000
zA = rng.standard_normal((n_mc, d))
zB = rng.standard_normal((n_mc, d))
gA = gf + np.sqrt(C) * zA                     # batch that defines the step
gB = gf + np.sqrt(C) * zB                     # fresh batch at the new point
# d/dalpha L_A(theta - alpha gA) = -<gA - alpha H gA, gA>  ;  fresh: -<gB - alpha H gA, gA>
# the zero of the expectation in alpha:
a_same_mc = np.mean(np.sum(gA * gA, 1)) / np.mean(np.sum(gA * lam * gA, 1))
a_fresh_mc = np.mean(np.sum(gB * gA, 1)) / np.mean(np.sum(gA * lam * gA, 1))
a_same = (gf @ gf + C.sum()) / (gf @ (lam * gf) + (lam * C).sum())
a_fresh = (gf @ gf) / (gf @ (lam * gf) + (lam * C).sum())
print(f"[7] greedy alpha*: same-batch MC {a_same_mc:.3f} vs formula {a_same:.3f};"
      f"  fresh-batch MC {a_fresh_mc:.3f} vs formula {a_fresh:.3f}")

# ---------------------------------------------------------------- 8
# stationary drift of the leaky-anchor hypergradient under constant-alpha SGD (start at optimum):
# E<g_t, S_t^beta> = - sum_i alpha lam_i c_i / ((2 - alpha lam_i)(1 - beta (1 - alpha lam_i)))
al = 0.5
Tm = 400_000
nrep = 1
zz = np.random.default_rng(1)
for beta in [0.0, 0.9, 0.99, 1.0]:
    th = np.zeros(d)
    S = np.zeros(d)
    acc, cnt = 0.0, 0
    for t in range(Tm):
        g = lam * th + np.sqrt(C) * zz.standard_normal(d)
        if t > 20_000:
            acc += g @ S
            cnt += 1
        S = beta * S + g
        th = th - al * g
    pred = -np.sum(al * lam * C / ((2 - al * lam) * (1 - beta * (1 - al * lam))))
    print(f"[8] beta={beta:<5} stationary E<g,S>: simulated {acc / cnt: .4e}   formula {pred: .4e}")

# ---------------------------------------------------------------- 9
# Lemma 5: gradient flow on (alpha, S) in theta = theta0 - alpha S conserves alpha^2/mu - |S|^2/eta
mu_, eta_, dt = 0.5, 2.0, 1e-4
a_, S_ = 0.2, np.zeros(d)
cons = lambda a, S: a * a / mu_ - S @ S / eta_
c0 = cons(a_, S_)
for _ in range(200_000):
    g = lam * (th0 - a_ * S_)
    a_, S_ = a_ + dt * mu_ * (g @ S_), S_ + dt * eta_ * a_ * g
print(f"[9] conservation alpha^2/mu - |S|^2/eta: start {c0:.5f} end {cons(a_, S_):.5f}; drift relative to "
      f"alpha^2/mu = {a_ * a_ / mu_:.2f}: {abs(cons(a_, S_) - c0) / (a_ * a_ / mu_):.1e} (Euler discretization)")
```

### B.2 Sensitivity variants (`tests/test_anchored_sensitivity.py`; PyTorch, float64)
* T1/T2: `full` and `v2` against finite differences (global α).
* T2b: per-block exactness under a block-diagonal Hessian.
* T3: v1 equals `HorizonHD(β=1, r=1)`.
* T5: Lemma 4 (the exact per-block v2 form; the cheap form is about 35% off in this example).

```python
"""Finite-difference verification of AnchoredSensitivityHD (float64, small tanh MLP, fixed batches)."""
import math
import torch
from anchored_sensitivity import AnchoredSensitivityHD
from horizon_hd import HorizonHD

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
D_IN, D_H, N, T = 5, 7, 16, 25
X = [torch.randn(N, D_IN) for _ in range(T + 1)]
Y = [torch.randn(N, 1) for _ in range(T + 1)]
W1_0 = torch.randn(D_H, D_IN) * 0.5
W2_0 = torch.randn(1, D_H) * 0.5


def loss_fn(W1, W2, t):
    return 0.5 * ((torch.tanh(X[t] @ W1.T) @ W2.T - Y[t]) ** 2).mean()


def grads_at(W1, W2, t, create_graph=False):
    W1 = W1.detach().requires_grad_(True) if not create_graph else W1
    W2 = W2.detach().requires_grad_(True) if not create_graph else W2
    return torch.autograd.grad(loss_fn(W1, W2, t), (W1, W2), create_graph=create_graph)


def unrolled_loss(alpha, t):
    """L_t(theta_t(alpha)) for the anchored trajectory with constant alpha (same alpha on both blocks
    is NOT assumed: alpha is a pair)."""
    a1, a2 = alpha
    S1, S2 = torch.zeros_like(W1_0), torch.zeros_like(W2_0)
    for i in range(t):
        g1, g2 = grads_at(W1_0 - a1 * S1, W2_0 - a2 * S2, i)
        S1, S2 = S1 + g1, S2 + g2
    return loss_fn(W1_0 - a1 * S1, W2_0 - a2 * S2, t), (S1, S2)


def run_opt(mode, steps, lr=0.3, meta_lr=0.0):
    W1 = torch.nn.Parameter(W1_0.clone())
    W2 = torch.nn.Parameter(W2_0.clone())
    opt = AnchoredSensitivityHD([W1, W2], lr=lr, meta_lr=meta_lr, mode=mode, lr_bounds=(1e-12, 1e12),
                                max_log_step=1e9)
    hs = []
    for t in range(steps):
        opt.step(lambda: loss_fn(W1, W2, t))
        hs.append(torch.stack(opt.last_h))
    return hs, (W1, W2)


# The one-HVP recursions are exact for a GLOBAL alpha (sum of per-block h = -dL/dalpha_common) and,
# per block, when the Hessian is block-diagonal. With per-block alphas and cross-block curvature they are
# the standard SMD-style "diagonal" approximation.
lr, eps = 0.3, 1e-6


def fd_common(t, F):
    return (F(lr + eps) - F(lr - eps)) / (2 * eps)


# --- T1: full mode, global alpha: sum_k h_k == -dL_t/dalpha of the constant-alpha unrolled trajectory
hs, _ = run_opt("full", T + 1, lr=lr)
worst = 0.0
for t in [1, 2, 5, 10, 25]:
    fd = fd_common(t, lambda a: unrolled_loss([a, a], t)[0])
    worst = max(worst, float(abs(hs[t].sum() + fd) / abs(fd)))
print(f"[T1] full sensitivity (global alpha) vs FD of unrolled anchored loss: max rel err {worst:.1e}")
assert worst < 1e-6

# --- T2: v2, global alpha: sum_k h_k == -d/dalpha L_t(theta0 - alpha (S_{t-1} + g_{t-1}(theta0 - alpha S_{t-1})))
hs2, _ = run_opt("v2", T + 1, lr=lr)
worst = 0.0
for t in [2, 5, 10, 25]:
    _, (S1p, S2p) = unrolled_loss([lr, lr], t - 1)                # S_{t-1}, detached constants

    def F(a):
        g1, g2 = grads_at(W1_0 - a * S1p, W2_0 - a * S2p, t - 1)
        return loss_fn(W1_0 - a * (S1p + g1), W2_0 - a * (S2p + g2), t)
    fd = fd_common(t, F)
    worst = max(worst, float(abs(hs2[t].sum() + fd) / abs(fd)))
print(f"[T2] v2 (global alpha) vs FD of the one-step formula:                   max rel err {worst:.1e}")
assert worst < 1e-6

# --- T2b: block-diagonal Hessian (separable loss) -> per-block h_k is the exact per-block derivative
def sep_loss(W1, W2, t):
    return 0.5 * ((torch.tanh(X[t] @ W1.T).sum(1, keepdim=True) - Y[t]) ** 2).mean() + \
           0.5 * ((W2 ** 2).sum() - 1.0) ** 2 * (1 + 0.1 * t % 3)


def sep_unrolled(al, t):
    S1, S2 = torch.zeros_like(W1_0), torch.zeros_like(W2_0)
    for i in range(t):
        a = (W1_0 - al[0] * S1).requires_grad_(True)
        b = (W2_0 - al[1] * S2).requires_grad_(True)
        g1, g2 = torch.autograd.grad(sep_loss(a, b, i), (a, b))
        S1, S2 = S1 + g1, S2 + g2
    return sep_loss(W1_0 - al[0] * S1, W2_0 - al[1] * S2, t)


W1 = torch.nn.Parameter(W1_0.clone())
W2 = torch.nn.Parameter(W2_0.clone())
opt = AnchoredSensitivityHD([W1, W2], lr=0.0, meta_lr=0.0, mode="full", lr_bounds=(1e-12, 1e12))
opt.param_groups[0]["lr"] = 1.0
alphas = [0.3, 0.05]
for p, a0 in zip([W1, W2], alphas):
    opt.state[p]["theta0"] = p.detach().clone()
    opt.state[p]["S"] = torch.zeros_like(p)
    opt.state[p]["q"] = torch.zeros_like(p)
    opt.state[p]["log_lr"] = torch.tensor(math.log(a0))
hsb = []
for t in range(11):
    opt.step(lambda: sep_loss(W1, W2, t))
    hsb.append(torch.stack(opt.last_h))
worst = 0.0
for t in [3, 10]:
    for k in range(2):
        ap, am = list(alphas), list(alphas)
        ap[k] += eps
        am[k] -= eps
        fd = (sep_unrolled(ap, t) - sep_unrolled(am, t)) / (2 * eps)
        worst = max(worst, float(abs(hsb[t][k] + fd) / abs(fd)))
print(f"[T2b] full sensitivity, separate alphas, block-diagonal H, per block: max rel err {worst:.1e}")
assert worst < 1e-6

# --- T3: v1 with meta-learning on == HorizonHD(beta=1, retro=1, norm='cos') trajectory
_, (A1, A2) = run_opt("v1", T, lr=0.1, meta_lr=0.05)
W1 = torch.nn.Parameter(W1_0.clone())
W2 = torch.nn.Parameter(W2_0.clone())
opt = HorizonHD([W1, W2], lr=0.1, meta_lr=0.05, beta=1.0, retro=1.0, norm="cos",
                lr_bounds=(1e-12, 1e12), max_log_step=1e9)
for t in range(T):
    opt.zero_grad()
    loss_fn(W1, W2, t).backward()
    opt.step()
err = max((A1 - W1).abs().max().item(), (A2 - W2).abs().max().item())
print(f"[T3] v1 == HorizonHD(beta=1, retro): max abs diff {err:.1e}")
assert err < 1e-12

# --- T4: with meta-learning on, v2 and full run and stay finite
for mode in ["v2", "full"]:
    _, (B1, B2) = run_opt(mode, T, lr=0.1, meta_lr=0.05)
    assert torch.isfinite(B1).all() and torch.isfinite(B2).all()
print("[T4] v2/full with meta-learning: finite")

# --- T5 (Lemma 4): exact per-block v2 with separate alphas and cross-block curvature:
#     dL_t/dalpha_k = -<g_t^k, S_t^k> + < [H_{t-1} A g_t]^k , S_{t-1}^k >      (A = blockdiag(alpha_j I))
#     The cheap "concatenated" form alpha_k <g_t^k, [H_{t-1} S_{t-1}]^k> is only an approximation.
S1p, S2p = torch.randn_like(W1_0) * .3, torch.randn_like(W2_0) * .3        # S_{t-1}, constants
al = [0.4, 0.15]


def F2(a1, a2):
    A_ = (W1_0 - a1 * S1p).requires_grad_(True)
    B_ = (W2_0 - a2 * S2p).requires_grad_(True)
    g1, g2 = torch.autograd.grad(loss_fn(A_, B_, 3), (A_, B_))
    return loss_fn(W1_0 - a1 * (S1p + g1), W2_0 - a2 * (S2p + g2), 4)


fd = [(F2(al[0] + eps, al[1]) - F2(al[0] - eps, al[1])) / (2 * eps),
      (F2(al[0], al[1] + eps) - F2(al[0], al[1] - eps)) / (2 * eps)]
A_ = (W1_0 - al[0] * S1p).requires_grad_(True)
B_ = (W2_0 - al[1] * S2p).requires_grad_(True)
g1p, g2p = torch.autograd.grad(loss_fn(A_, B_, 3), (A_, B_), create_graph=True)
St1, St2 = S1p + g1p.detach(), S2p + g2p.detach()
C_ = (W1_0 - al[0] * St1).requires_grad_(True)
D_ = (W2_0 - al[1] * St2).requires_grad_(True)
gt1, gt2 = torch.autograd.grad(loss_fn(C_, D_, 4), (C_, D_))
HAg = torch.autograd.grad((g1p, g2p), (A_, B_), grad_outputs=(al[0] * gt1, al[1] * gt2), retain_graph=True)
exact = [-(gt1 * St1).sum() + (HAg[0] * S1p).sum(), -(gt2 * St2).sum() + (HAg[1] * S2p).sum()]
HS = torch.autograd.grad((g1p, g2p), (A_, B_), grad_outputs=(S1p, S2p))
cheap = [-(gt1 * St1).sum() + al[0] * (gt1 * HS[0]).sum(), -(gt2 * St2).sum() + al[1] * (gt2 * HS[1]).sum()]
worst = max(float(abs(exact[k] - fd[k]) / abs(fd[k])) for k in range(2))
gap = max(float(abs(cheap[k] - fd[k]) / abs(fd[k])) for k in range(2))
print(f"[T5] exact per-block v2 (H(Ag) form) vs FD: max rel err {worst:.1e}; cheap form rel err {gap:.1e}")
assert worst < 1e-6

print("OK")
```

### B.3 Tests you still have to write
1. **Fast vs reference.** Every fast optimizer matches Appendix A in float64 to ≤ 1e-10 over 300+ steps, for all modes (β ∈ {0, 0.9, 0.99, 1}, r ∈ {0, 1}, cos/rms, tensor/row/group, momentum on/off), on (a) a block-heterogeneous noisy quadratic (Appendix C), (b) a small MLP, and (c) a 2-layer transformer with math SDPA.
2. **Degenerate cases.**
   * lr = 0 leaves parameters unchanged.
   * meta_lr = 0 equals SGD (or SGD-M) with LR α₀, for every β and r.
   * β = 0 equals a direct implementation of Baydin's multiplicative HD.
3. **Finite-difference HVP vs autograd HVP** (math SDPA) on a tiny transformer, in fp32 and under bf16 autocast. Record the relative error and choose ε.
4. **Resume.** N steps, then checkpoint, then resume for M steps, compared with N + M uninterrupted (bitwise in fp32 on one GPU, or ≤ 1e-6).
5. **DDP.** 2-GPU DDP equals 1 GPU with the same global batch (tolerance); α is identical across ranks.
6. **bf16 vs fp32.** A short proxy run under bf16 autocast tracks the fp32 run closely, and the α trajectories correlate above 0.99.
7. **Overhead.** Optimizer step time ≤ 5% (first-order) of total step time at the M scale; report the HVP variants' cost.

---

## Appendix C: Block-heterogeneous noisy quadratic model (unit and mechanism testbed; GPU, seconds)

Use this for fast tests and for checking mechanism predictions (Lemma 7, P1/P2, H5) before transformer runs. **It is not evidence for the paper.**

* $K$ blocks of dimension $d_k$, with $f(\theta)=\tfrac12\sum_k\theta_k^\top\mathrm{diag}(\lambda_k)\theta_k$, optimum at 0.
* Block curvature scales spanning orders of magnitude (e.g. $\lambda_{\max}^{(k)}\in\{1,10^{-1},10^{-2},10^{-3}\}$, condition number about 10 inside each block).
* Minibatch noise $\xi\sim\mathcal N(0,H/B)$ (Zhang et al., 2019 NQM style).
* Initialize so every block starts with equal loss.
* All per-block scalar methods are invariant to rotations inside a block, so a diagonal $H$ is without loss of generality for them.
* Vectorize over a hyperparameter grid and over seeds: tensors shaped (grid, seeds, K, d).
* Oracle reference: SGD with per-block LR $c/\lambda_{\max}^{(k)}$, with $c$ tuned per block, constant or linear-decay.
* Deterministic variant ($B=\infty$) for checking the edge-of-stability prediction ($\alpha_k\lambda_{\max}^{(k)}\to[1,2]$).

---

## Appendix D: Notes from the owner's ChatGPT discussion (incorporated)

The owner's brainstorming transcripts with ChatGPT were analyzed and merged into the main text. What came from them, and the corrections made along the way:

**Incorporated** (re-derived; identities checked with the Appendix B tooling, or by hand where trivial):
* The precise reading of the algorithm ($S_0=0$; $h_t$ from the forward pass that used $S_t$) and the ordering-bias warning (§3.2, §12).
* The autograd construction `w = anchor - alpha * S.detach()`, $\nabla_{\rm anchor}L=\nabla_wL$ (identity Jacobian), and the log-parameter reading $\partial L/\partial s=\langle g,\theta_t-\theta_0\rangle$ (§3.2).
* The **dual-averaging** interpretation and the MDA / D-Adaptation / Xiao prior work (§3.3b, §4). This is the most important addition for positioning.
* The per-block preconditioner $P_{k,t}=\alpha_{k,t+1}I+\mu S^{(k)}_tS^{(k)\top}_t$ with its two eigenvalues, the same-loss descent condition $L\lambda_{\max}(P)<2$, the linearized coupled stability condition $\lambda(\alpha^\star+\mu(S^\star)^2)<2$, the α-subproblem curvature $S^\top HS$ with cross-block coupling $S^{(k)\top}H_{kj}S^{(j)}$, and the normalized meta-step $\gamma\langle g,S\rangle/(\|S\|^2+\varepsilon)$ whose extra eigenvalue is $\le\gamma$ (§3.4, §3.7).
* Buffer-update analysis: accumulation equals GD on $S$ preconditioned by $1/\alpha_t$; Euclidean GD gives an effective LR $\gamma\alpha^2$; the $\gamma\alpha^p$ ablation family; compare at matched effective LR (§3.8).
* The one-step HVP hypergradient (the owner's derivation): its sign, its exact meaning (a one-step surrogate with a persistent shared α), the Level 0 / 1 / K differentiation axis, the lookahead vs current-loss organization and its memory implications, the exact per-block form via one HVP of $D_\alpha g_t$ (matches Lemma 4), and the advice not to substitute Adam's $v$ or Hutchinson diagonals for $H$ (§3.6).
* Why the retroactive form is what makes α differentiable from the current loss: the prospective $D_{t+1}=D_t+\alpha_tg_t$ variant has zero current-loss derivative (§3.2).
* The owner's motivation of avoiding explicit weight updates, with the analysis that it is not a compute or memory saving (§2, §5).
* Held-out-loss (bilevel) hypergradients as an exploratory arm (§3.16, A12, H13).
* Diagnostics: the retroactive-share ratio $R_{k,t}$, the curvature-term ratio and sign-flip frequency (§3.6, §10).

**Corrections and clarifications recorded during that discussion:**
* The curvature correction is $+\alpha H_{t-1}S_{t-1}$ in $d\theta_t/d\alpha$ (not $-$).
* The owner's $f(aw+b)$ example does not imply a different anchor gradient: the anchor enters additively (identity Jacobian); the $-\alpha$ scaling appears only for derivatives with respect to $S$.
* The cheap block-restricted curvature term is an approximation when rates are per block; the exact form needs the full HVP $H_{t-1}D_\alpha g_t$ (verified in T5).

**Where this brief goes beyond the transcripts:** the horizon family and its exact equivalences (§3.5, §3.12), the fresh- vs same-batch analysis of HD's lookahead (§3.9), the noise-floor drift law (§3.10), the dilution/amplification predictions (§3.11), the exact full-horizon recursion with verified code (§3.6, Appendix A), the 2026 literature (SGD-LL and others), and the experimental program.

<!-- The owner may paste additional notes below this line. Agent: reconcile them with §3 and log any differences. -->
