# TensorCatalog: Latent Reasoning at Zero Decoding Cost through Hidden State Recycling

## Abstract

Standard autoregressive decoding links successive steps only through sampled discrete tokens, discarding the continuous semantics encoded in previous hidden states. In contrast, recurrent multi-token prediction (MTP) heads propagate both token embeddings and hidden states across steps. Remarkably, an MTP head with only $\sim 2\%$ of the backbone parameters achieves a loss comparable to that of a standalone model $\sim 25\%$ its size, underscoring the value of continuous hidden state feedback. Motivated by this observation, we introduce Recycler, a simple and effective architecture that feeds both embedding and the previous token's hidden state through a lightweight adapter into the next step, preserving richer cross-step semantics with negligible decoding overhead. We develop a complete training and inference pipeline covering pretraining, SFT, continued pretraining and RL. Experiments demonstrate strong parameter efficiency, particularly in reasoning and mathematics. Although earlier integration yields larger gains, Recycler remains effective when introduced at later stages, outperforming models with $2\text{–}3\times$ more parameters.

## 1 Introduction

Modern language models [3-6, 18-20, 42] generate text autoregressively. Each decoding step produces a top-layer hidden state $\boldsymbol{h}\_t$ processed by the full Transformer [51] stack, yet only a sampled discrete token $x\_{t+1}$ is directly passed to the next step. This lossy discretization discards rich information encoded in $\boldsymbol{h}\_t$. Preserving this state suggests a complementary computational scaling axis that reuses computation across decoding steps, rather than relying only on more parameters or training tokens.

An empirical clue supporting this direction comes from recurrent multi-token prediction (MTP) [10, 22]. A recurrent MTP module resembles a standard autoregressive language model, but additionally receives the backbone's previous-position hidden state alongside the token embedding. With only about $2\%$ additional parameters, it achieves a loss comparable to that of a standalone model one quarter the size of the backbone. This observation motivates our hypothesis that the backbone itself can benefit from recycling its previous output hidden state. Building on this observation, we introduce Recycler, a simple architecture that augments the current token embedding with the previous token's top-layer hidden state through a lightweight adapter. Whereas standard autoregressive decoding compresses the latest top-layer output into a sampled discrete token before the next step, Recycler retains this token path while adding a parallel continuous path that directly carries the full-stack hidden state forward, as illustrated in Figure 2. This recycle branch bypasses the discrete bottleneck and creates a continuous recurrent computation whose effective depth grows with the generated sequence. Notably, since the previous hidden state is already available and the adapter is lightweight, Recycler has approximately the same autoregressive decoding cost as a standard Transformer.

Although Recycler is straightforward during autoregressive inference, its sequential dependency prevents exact one-pass parallel training. We address this challenge with a multi-pass approximation, where each pass uses the recycled hidden states from the preceding pass while retaining parallelism across sequence positions. We empirically show that the resulting model generalizes to autoregressive recurrence while maintaining a decoding cost approximately equal to that of a standard Transformer.

Beyond the architecture itself, we develop an end-to-end framework that supports Recycler throughout model training and deployment. The framework covers pretraining, continued pretraining, supervised fine-tuning, reinforcement learning. Experiments validate Recycler throughout the entire pipeline. It consistently improves pretraining and downstream performance, with particularly pronounced gains on mathematics and reasoning tasks. These benefits persist when Recycler is introduced only during continued pretraining or supervised fine-tuning, and are further strengthened through reinforcement learning. Finally, we resolve Recycler's parallel-verification challenge in MTP-based speculative decoding, enabling further inference acceleration.

## 2 Approach

### 2.1 Architecture

As shown in Figure 2 (a), during standard autoregressive decoding, token $x\_t$ is mapped to its embedding and fed into the Transformer backbone $\mathcal{T}$ to produce the top-layer hidden state $\boldsymbol{h}\_t$. The hidden state is then projected by the unembedding layer, from which the next token $x\_{t+1}$ is sampled and re-embedded as the input to the next decoding step. Let $\Phi$ denote this stochastic discrete feedback operation, consisting of unembedding, softmax normalization, token sampling, and re-embedding. The relationship between two consecutive hidden states can then be written compactly as

$$
\boldsymbol{h}\_{t+1} = \mathcal{T}\big(\Phi(\boldsymbol{h}\_t)\big). \tag{1}
$$

The discrete feedback operator $\Phi$ forms an information bottleneck: it maps a continuous hidden state carrying rich contextual information to a discrete token ID before reconstructing an input embedding. Our key observation is that this bottleneck is not necessary for propagating information across autoregressive decoding steps, because the previous hidden state has already been computed and remains available. Recycling that state provides a richer input to the next step without requiring an additional backbone forward pass, incurring only the negligible cost of a lightweight adapter.

As shown in Figure 2(b), Recycler introduces a continuous feedback pathway alongside the lossy discrete operator $\Phi$. This pathway transforms the previous hidden state with a lightweight adapter $\mathcal{A}$ composed of RMS normalization and a learnable linear projection. It therefore preserves substantially more of the information in the hidden state than the discrete feedback pathway. We define the $\oplus$ operator in Figure 2(b) as

$$
\begin{aligned}
\mathcal{A}(\boldsymbol{h}) &:= \mathrm{Linear}\big(\mathrm{RMSNorm}(\boldsymbol{h})\big), \\
x \oplus \boldsymbol{h} &:= \mathrm{Emb}(x) + \mathcal{A}(\boldsymbol{h}).
\end{aligned}
\tag{2}
$$

The relationship between consecutive hidden states in Recycler then becomes

$$
\boldsymbol{h}\_{t+1} = \mathcal{T}\big(x\_{t+1} \oplus \boldsymbol{h}\_t\big) = \mathcal{T}\big(\Phi(\boldsymbol{h}\_t) + \mathcal{A}(\boldsymbol{h}\_t)\big). \tag{3}
$$

**(a) Standard Decoding**

**(b) Recycler Decoding**

**Legend:** Token $x\_j$ | Hidden State $\boldsymbol{h}\_j$ | $(k)$ Recycle Step | $\oplus$ Fuse Operator | Unembed | Hidden Recycle

**Figure 2** (a) Decoding process of the standard model, (b) decoding process of the Recycler model, and (c) training and prefill procedure of the Recycler model.

Unlike the standard model, whose cross-step communication is restricted to the discrete token pathway, Recycler therefore propagates continuous hidden-state information recurrently across decoding steps.

### 2.2 Recycler Decoding, Training and Prefill

We next describe how the recurrent Recycler architecture is executed during decoding and approximated efficiently during training and prompt prefilling.

#### 2.2.1 Decoding

At generation time, Eq. (3) is applied recursively after each sampled token: the hidden state from the current step is adapted and combined with the next token embedding to produce the next hidden state. We write $\boldsymbol{h}\_t^{(k\_{\mathrm{dec}})}$, where the superscript $k\_{\mathrm{dec}}$ denotes the effective **decoding recycle depth**. At depth $k\_{\mathrm{dec}}$, the current hidden state can incorporate continuous information propagated through the most recent $k\_{\mathrm{dec}}$ positions. This recurrence is not truncated during autoregressive decoding, so $k\_{\mathrm{dec}}$ grows with the generated sequence length. This creates an effectively unbounded-depth information pathway without requiring additional backbone passes per generated token.

#### 2.2.2 Training

Exact training under Eq. (3) would introduce a sequential hidden-state dependency across token positions, preventing the standard token-parallel training of Transformers. We instead truncate the recycle depth to a fixed training depth $k\_{\mathrm{train}}$ and approximate the recurrence using $k\_{\mathrm{train}}$ sequence-level forward passes:

$$
\begin{aligned}
\boldsymbol{h}\_t^{(1)} &= \mathcal{T}\big(\mathrm{Emb}(x\_t)\big), \\
\boldsymbol{h}\_t^{(i)} &= \mathcal{T}\big(\mathrm{Emb}(x\_t) + \mathcal{A}(\boldsymbol{h}\_{t-1}^{(i-1)})\big), \\
\mathcal{L}^{(i)} &= \sum\_t \mathrm{NTP}\big(\boldsymbol{h}\_t^{(i)}, x\_{t+1}\big), \qquad
\mathcal{L} = \sum\_{i=1}^{k\_{\mathrm{train}}} \mathcal{L}^{(i)}.
\end{aligned}
\tag{4}
$$

Here, $\boldsymbol{h}\_t^{(i)}$ is the hidden state at position $t$ and recycle depth $i$, and NTP denotes the next-token cross-entropy loss. Given all states from pass $i-1$, all token positions in pass $i$ can be computed in parallel. Thus, the method preserves token-level parallelism while allowing $\boldsymbol{h}\_t^{(k\_{\mathrm{train}})}$ to incorporate continuous information propagated through the most recent $k\_{\mathrm{train}}$ positions. When $k\_{\mathrm{train}} \geq t$ (assuming one-indexed positions), it recovers the exact recurrent state in Eq. (3).

We use three implementation choices to stabilize this approximation:

- We zero-initialize the linear projection in $\mathcal{A}$, making $\mathcal{A}(\cdot) = 0$ at initialization and recovering the standard Transformer computation.
- We apply the NTP loss to all $k\_{\mathrm{train}}$ passes, which aligns the representation spaces of $\boldsymbol{h}\_t^{(1)}, \ldots, \boldsymbol{h}\_t^{(k\_{\mathrm{train}})}$ and reduces the train–decode distribution gap.
- We tie the input embedding and output projection weights to better align their representation spaces, reducing the mismatch between successive recycle depths.

#### 2.2.3 Prefill

During prompt prefilling, reproducing Eq. (3) token by token would match autoregressive decoding exactly, but would eliminate the sequence parallelism of standard Transformer prefilling. We therefore use parallel forward passes over the full prompt and consider two operating points:

- **Prefill $k\_{\mathrm{prefill}} = 2$.** We apply two parallel passes over the prompt and initialize decoding with the final prompt-token state from the second pass. This doubles prefill computation relative to a standard model.
- **Prefill $k\_{\mathrm{prefill}} = 1$.** We perform one standard parallel pass over the prompt and initialize decoding with its final token state. This matches the prefill computation of a standard model.

Both strategies subsequently use the same recurrent decoding procedure in which the effective recycle depth grows with the generated sequence length.

### 2.3 Recycler Across the LLM Lifecycle

Building on the decoding recurrence, parallel training approximation, and prefill strategies introduced above, we now describe how Recycler is integrated into each stage of the LLM lifecycle. Specifically, we detail its stage-specific implementation for pretraining, supervised fine-tuning, reinforcement learning.

#### 2.3.1 Pretraining

We apply the parallel approximation from Section 2.2.2 with $k\_{\mathrm{train}} = 2$ to the full pretraining sequence and supervise both recycle depths with the NTP objective. Because pretraining examples are treated as continuous token sequences, there is no separate prompt-prefill region during optimization.

**Extrapolation Analysis.** Considering that Recycler is trained using an approximate algorithm, its inference procedure is not fully aligned with the training process. This discrepancy raises a natural concern that the approximation used during training might not generalize well to the decoding stage. We first conduct a validation experiment to examine this: we train a tiny Recycler model (MoE architecture, 151M activated parameters, 1B total parameters) with $k\_{\mathrm{train}} = 2$ for 200B tokens. During inference, we vary $k\_{\mathrm{dec}}$ from 1 to 32 to assess whether performance remains stable beyond the training depth. The results are shown in Figure 3:

**Figure 3** Extrapolation analysis under increasing decoding recycle depth $k\_{\mathrm{dec}}$. The model is trained with $k\_{\mathrm{train}} = 2$ and evaluated with $k\_{\mathrm{dec}} \in [1, 32]$ on an i.i.d. evaluation set and the downstream benchmarks MMLU and C-Eval.

- When $k\_{\mathrm{dec}} = 1$, the model already achieves substantial gains over the baseline, indicating that even without using the hidden state of the previous token, this training practice of incorporating future token loss into consideration still yields performance benefits.
- When $k\_{\mathrm{dec}} > 1$, the Recycler model achieves further gains and remains stable across subsequent extrapolation steps. This indicates that the training-inference inconsistency in Recycler can be naturally mitigated through extrapolative generalization.

The training cost of Recycler is $2\times$ that of the standard model. Through the extrapolation experiments above, we have preliminarily demonstrated the effectiveness of our approximate training method. However, as the model size and data volume grow, extrapolation issues may arise and lead to a higher evaluation loss, as illustrated in Figure 4. To address this, we perform a third forward pass on a subset of steps during training. Compared to Recycler training with two forward passes, this approach incurs only an additional $3.3\%$ in training cost while fully resolving the extrapolation problem of Recycler.

**Figure 4** Evaluation loss under increasing $k\_{\mathrm{dec}}$ for the original $k\_{\mathrm{train}} = 2$ recipe and a partial $k\_{\mathrm{train}} = 3$ recipe with additional $3.3\%$ training cost.

#### 2.3.2 Supervised Fine-Tuning

SFT mainly requires aligning response training with the prompt-prefill strategy used at inference. Response tokens follow the parallel training approximation in Section 2.2.2 with $k\_{\mathrm{train}} = 2$, and the NTP loss is masked to the response region. For $k\_{\mathrm{prefill}} = 2$, both prompt passes retain the recycled-hidden pathway. For $k\_{\mathrm{prefill}} = 1$, the second pass is still required to train response tokens, but the recycled contribution $\mathcal{A}(\boldsymbol{h}\_{t-1}^{(1)})$ is set to zero at prompt positions to match single-pass prefilling.

Generally speaking, models trained and inferred with prefill $k\_{\mathrm{prefill}} = 2$ exhibit markedly better performance on downstream metrics than those using prefill $k\_{\mathrm{prefill}} = 1$.

#### 2.3.3 Reinforcement Learning

For RL, the main adaptation lies in the rollout process: the prompt follows the selected prefill strategy, whereas the response is generated through recurrent Recycler decoding.

To further stabilize training, we adopt GSPO [65] as our reinforcement learning algorithm:

$$
\mathcal{J}\_{\mathrm{GSPO}}(\theta)
= \mathbb{E}\_{x \sim \mathcal{D},\, \{y\_i\}\_{i=1}^{G} \sim \pi\_{\theta\_{\mathrm{old}}}(\cdot \mid x)}
\left[ \frac{1}{G} \sum\_{i=1}^{G} \min\Big( s\_i(\theta)\, \hat{A}\_i,\; \mathrm{clip}\big(s\_i(\theta),\, 1 - \varepsilon\_{\mathrm{low}},\, 1 + \varepsilon\_{\mathrm{high}}\big)\, \hat{A}\_i \Big) \right],
\tag{5}
$$

where $G$ is the number of generated responses to each query $x$, and the group-based advantage estimation is:

$$
\hat{A}\_i = \frac{r(x, y\_i) - \mathrm{mean}\big(\{r(x, y\_j)\}\_{j=1}^{G}\big)}{\mathrm{std}\big(\{r(x, y\_j)\}\_{j=1}^{G}\big)}, \tag{6}
$$

and the importance ratio $s\_i(\theta)$ is defined on sequence level, which addresses the high-variance issue of the importance ratio in GRPO:

$$
s\_i(\theta)
= \left( \frac{\pi\_{\theta}(y\_i \mid x)}{\pi\_{\theta\_{\mathrm{old}}}(y\_i \mid x)} \right)^{\frac{1}{|y\_i|}}
= \exp\left( \frac{1}{|y\_i|} \sum\_{t=1}^{|y\_i|} \log \frac{\pi\_{\theta}(y\_{i,t} \mid x, y\_{i,<t})}{\pi\_{\theta\_{\mathrm{old}}}(y\_{i,t} \mid x, y\_{i,<t})} \right). \tag{7}
$$

As we demonstrate later, under an equal RL training budget, the Recycler architecture yields greater score gains than the standard model.

## 3 Experiments

We evaluate Recycler throughout the model-development pipeline, covering pre-training (PT), continued pre-training (CT), supervised fine-tuning (SFT), reinforcement learning (RL). All Recycler models in this section are trained with a recycle depth of $k = 2$. At inference time, the recurrent hidden-state pathway is rolled out autoregressively for as many decoding steps as required by each evaluation, thereby extrapolating beyond $k = 2$. Prompt prefill is treated separately. For selected evaluations, we report both prefill $\times 1$ which recycle step $k = 1$ and prefill $\times 2$ which recycle step $k = 2$: the former performs one parallel forward pass over the prompt, whereas the latter performs two and initializes decoding from the final hidden state of the second pass. Both modes subsequently use the same recurrent Recycler decoding process.

For comparisons designed to be controlled, the baseline and Recycler share the same backbone, data mixture, optimization recipe, and training budget, with the Recycler pathway as the intended architectural difference. Unless otherwise specified, models evaluated after CT use the CT-introduced configuration, in which Recycler is activated during continued pre-training and retained through SFT and RL. Section 3.2 examines why this configuration offers a favorable balance between cumulative training cost and final performance.

### 3.1 Pre-Training

**Setup.** We validate Recycler on two independently trained model versions, denoted $\text{model}\_\alpha$ and $\text{model}\_\beta$. $\text{model}\_\alpha$ provides a controlled comparison between Recycler and its baseline, allowing us to examine both optimization behavior and the evolution of downstream capabilities. $\text{model}\_\beta$ provides a complementary scaling comparison.

**Token Efficiency.** As shown in Figure 5, Recycler exhibits markedly lower training and validation loss curve than the baseline. At the end of the training, Recycler achieves approximately $0.02$ gains in training loss. On the held-out validation set, the first forward ($L^{(1)}$) already achieves a reduction of approximately $0.013$, while the recycled second forward ($L^{(2)}$) enlarges the reduction to approximately $0.033$. This indicates that Recycler converges faster and achieves greater token efficiency. On the other hand, this also indicates that, when the training cost of the two forward passes is not taken into account, Recycler can achieve better performance with the same number of training tokens.

**Figure 5** $\text{model}\_\alpha$ training and validation loss for Baseline and Recycler in the pre-training stage. The $L^{(1)}$ and $L^{(2)}$ indicate the loss curves with one and two forward passes, respectively.

**Gains across Benchmark Sources.** We evaluate intermediate $\text{model}\_\alpha$ checkpoints on 14 datasets covering knowledge and comprehensive understanding (MMLU [28], MMLU-Pro [54], C-Eval [30], GPQA [45], TriviaQA [34], SimpleQA [56], and ChineseSimpleQA [27]), logical reasoning (AGI-Eval [66], DROP [13], ARC-AGI [9], and BBH [49]), code generation (HumanEval [7] and MBPP+ [1, 38]), and mathematics (MATH [29]).

At the end of the training, Recycler $\text{model}\_\alpha$ outperforms the baseline on all 14 datasets. The downstream average rises from 42.97 to 46.21, an absolute gain of 3.24. The largest gains occur on DROP (+6.31 points), BBH (+5.53), AGI-Eval (+5.35), MBPP+ (+4.76), and MMLU-Pro (+4.40), while knowledge-heavy benchmarks show smaller but still positive improvements. This source-level consistency indicates that the loss reduction in Figure 5 transfers broadly to downstream capabilities, with particularly strong effects on reasoning and structured generation. To further validate the effectiveness of Recycler, we additionally evaluate recycle step $k = 1$. As shown in Figure 6, even when the prompt is processed with only a single forward pass—the same prefill procedure as standard autoregressive inference—Recycler yields clear improvements over the baseline on most evaluation benchmarks.

**Comparison with a $2.4\times$ Larger Baseline.** We next evaluate $\text{model}\_\beta$ to test whether Recycler's additional latent computation can substitute for model scaling. Figure 7 compares three systems: the $\text{model}\_\beta$ baseline, a baseline with $2.4\times$ the model size (Baseline $\times 2.4$), and the same-size Recycler. Recycler retains the same token-by-token decoding cost and KV-cache footprint as its same-size baseline, whereas Baseline $\times 2.4$ increases the cost of every decoding step with model size.

The $\text{model}\_\beta$ results reproduce the broad source-level gains observed with $\text{model}\_\alpha$. Over the available overlap, Recycler improves substantially over the same-size baseline across the benchmark suite. More importantly, at the end of the displayed training window, Recycler reaches a downstream average of 64.76, compared with 66.43 for Baseline $\times 2.4$—a gap of only 1.68 despite the $2.4\times$ difference in model size. Recycler is close to the larger baseline on many individual sources and remains competitive across knowledge, reasoning, mathematics, and code benchmarks. The same-size baseline has a shorter available pre-training trajectory, ending near $50\%$ of the shared display range; we therefore show it without extrapolation and do not treat the three endpoints as an equal-training-budget comparison. The central comparison is instead computational: Recycler nearly matches the substantially larger model while preserving the decoding cost of the smaller model.

**Figure 6** $\text{model}\_\alpha$ benchmark trajectories over normalized pre-training progress. Markers denote observed checkpoints; no score smoothing, resampling, or imputation is applied. The bottom-right panel is the equal-weight average of the 14 datasets, computed at checkpoints shared by all datasets within each evaluation path. Recycler ($k = 1$) is shown only over its available early-training range.

**Figure 7** $\text{model}\_\beta$ benchmark trajectories for the same-size Baseline, Baseline $\times 2.4$, and Recycler over a common normalized training-progress reference. Markers are the recorded checkpoints; no smoothing, interpolation, or score-based filtering is applied. The bottom-right panel is the equal-weight average over the 14 benchmarks. The shorter same-size baseline trajectory is shown only over its available pre-training coverage and is not extrapolated.

**Summary.** The results consistently validate the effectiveness of the Recycler mechanism. Recycling hidden states provides an additional latent-computation step that lowers both training and validation loss, and the resulting gains transfer across all 14 downstream benchmark sources, with particularly strong improvements on reasoning and structured-generation tasks. Moreover, Recycler approaches the performance of a $2.4\times$ larger baseline while retaining the smaller model's autoregressive decoding cost and KV-cache footprint. These findings show that Recycler uses additional latent computation to improve optimization and downstream capability without requiring a proportional increase in model size or token-by-token decoding cost.

### 3.2 Continued Pre-Training and Supervised Fine-Tuning

#### 3.2.1 Training Efficiency

A central design question is when to introduce Recycler in a multi-stage training pipeline. We therefore compare three routes that activate Recycler at pre-training (PT), continued pre-training (CT), or supervised fine-tuning (SFT), respectively. Figure 8 summarizes the resulting cost-performance trade-off. Because the PT/CT and SFT results are reported on different benchmark suites, we restrict the analysis to the paired Baseline-Recycler comparison within each suite rather than comparing absolute scores across suites.

The paired comparisons exhibit a consistent pattern: enabling Recycler improves benchmark performance regardless of whether it is introduced at PT, CT, or SFT. This indicates that the mechanism does not depend on being present from the beginning of training; it can also be integrated into an existing model at a later stage and still yield a stable improvement over the corresponding baseline.

The introduction stage nevertheless has a substantial effect on training efficiency. Activating Recycler from PT changes the training paradigm for the entire pre-training trajectory. Although this route provides the longest exposure to recycled hidden states and achieves a strong final result, the additional computation is incurred throughout the most expensive stage of model development, making its cumulative training overhead prohibitively high. Introducing Recycler only at SFT lies at the opposite extreme: it requires little additional compute and still improves the paired baseline, but leaves a comparatively short adaptation window for the new recurrent information path.

Introducing Recycler at CT provides the most favorable compromise. It avoids retrofitting the additional training computation across the full PT budget, while retaining a sufficiently long continuation phase for the model to adapt to the modified training objective, fully optimize the recycled-hidden pathway, and converge before downstream post-training. The CT-introduced route therefore preserves a clear performance benefit without incurring the dominant cost of PT-time adoption. Based on this balance between effectiveness, cumulative training compute, and optimization horizon, we use the CT-introduced configuration in the subsequent SFT and RL experiments.

**Figure 8** Benchmark score versus estimated cumulative training compute for introducing Recycler at different stages. Each arrow connects a stage-matched baseline to the corresponding Recycler route within the same benchmark suite. Across PT-, CT-, and SFT-introduced settings, Recycler consistently moves the paired result toward higher benchmark performance. The PT/CT and SFT suites are shown separately because their absolute scores are not directly comparable.

#### 3.2.2 Supervised Fine-Tuning Evaluation

We further perform supervised fine-tuning (SFT) using the CT-stage checkpoints described above and summarize the results on 22 benchmarks in Table 1. All Recycler SFT variants are initialized from the CT-trained Recycler checkpoint. We evaluate two matched SFT-inference configurations in which the recycle depth is held consistent across the two stages: *prefill recycle step $k = 1$* uses $k = 1$ during both SFT and inference, whereas *prefill recycle step $k = 2$* uses $k = 2$ during both stages. The auxiliary $k = 2$-trained/$k = 1$-inference result is included only as a reference point, evaluating the same $k = 2$-trained checkpoint with a single prefill recycle step at inference.

The 22 benchmarks cover mathematics (AIME24 [41], AIME25 [41], OlympiadBench [26], OmniMath [16], HARP [61]), reasoning (ARC-AGI [9], KORBench [40], ProcBench [15], ZebraLogicBench [37]), instruction following (IFEval [67], COLLIE [60]), coding (LiveCodeBench [32], SWE Verified [33]), and comprehensive evaluations (GPQA-Diamond [45], SuperGPQA [39], HLE [44], MMLU [28], MMMLU [43], MMLU-Pro [54], ArenaHard [36], SimpleQA [56], ChineseSimpleQA [27]).

**Results.** Recycler improves the aggregate SFT performance under both prefill settings, with prefill recycle step $k = 2$ providing the strongest endpoint. Relative to the baseline, its Avg. score increases from 62.9 to 66.7 (+3.8 points). It outperforms the baseline on 19 of the 22 benchmarks, and the complete collection shows gains in all seven ability groups. The largest ability-level improvements are in instruction following (+9.3) and reasoning (+7.3).

The per-benchmark results reinforce this pattern. With prefill recycle step $k = 2$, Recycler improves ProcBench by 8.2 points, ARC-AGI by 9.3, ZebraLogicBench by 6.0, and KORBench by 5.6. It also yields substantial gains on SuperGPQA (+9.6), AIME25 (+5.4), HARP (+5.0), SWE Verified (+4.2), and MMLU (+4.0).

**Table 1** SFT results (%) on 22 Open Benchmark datasets. Best displayed values are bold. Prefill recycle steps $k = 1$ and $k = 2$ use one and two prompt prefill passes, respectively. The $k = 2$-trained/$k = 1$-inference route evaluates the $k = 2$ checkpoint with one prefill recycle step.

| Benchmark                  | Baseline | Prefill recycle step $k = 1$ | Prefill recycle step $k = 2$ | $k = 2$-trained / $k = 1$-inference |
| -------------------------- | -------- | ---------------------------- | ---------------------------- | ----------------------------------- |
| **Mathematics**            |          |                              |                              |                                     |
| AIME24                     | 81.3     | 85.3                         | 84.0                         | 85.7                                |
| AIME25                     | 75.3     | 79.0                         | 80.7                         | 80.3                                |
| OlympiadBench              | 88.7     | 88.0                         | 89.3                         | 86.7                                |
| OmniMath                   | 64.7     | 65.3                         | 65.3                         | 66.7                                |
| HARP                       | 79.0     | 86.3                         | 84.0                         | 85.7                                |
| **Reasoning**              |          |                              |                              |                                     |
| ARC-AGI                    | 32.6     | 39.4                         | 41.9                         | 37.0                                |
| KORBench                   | 59.3     | 62.9                         | 64.9                         | 62.5                                |
| ProcBench                  | 59.8     | 62.6                         | 68.0                         | 63.4                                |
| ZebraLogicBench            | 80.5     | 83.5                         | 86.5                         | 81.5                                |
| **Instruction Following**  |          |                              |                              |                                     |
| IFEval                     | 86.5     | 83.9                         | 84.8                         | 84.1                                |
| COLLIE                     | 60.4     | 50.2                         | 70.9                         | 44.4                                |
| **Coding**                 |          |                              |                              |                                     |
| LiveCodeBench              | 57.3     | 63.4                         | 59.5                         | 59.5                                |
| SWE Verified               | 41.8     | 45.8                         | 46.0                         | 40.4                                |
| **Comprehensive Evalsets** |          |                              |                              |                                     |
| GPQA-Diamond               | 72.1     | 73.2                         | 74.8                         | 73.3                                |
| SuperGPQA                  | 50.4     | 56.0                         | 60.0                         | 56.8                                |
| HLE                        | 9.3      | 12.0                         | 10.7                         | 12.0                                |
| MMLU                       | 82.7     | 84.0                         | 86.7                         | 82.7                                |
| MMMLU                      | 78.7     | 80.7                         | 82.0                         | 77.3                                |
| MMLU-Pro                   | 82.4     | 82.5                         | 82.4                         | 82.0                                |
| ArenaHard                  | 71.0     | 77.4                         | 68.6                         | 77.6                                |
| SimpleQA                   | 13.7     | 16.7                         | 15.3                         | 13.3                                |
| ChineseSimpleQA            | 57.3     | 57.7                         | 60.3                         | 53.3                                |
| **Avg.**                   | 62.9     | 65.3                         | 66.7                         | 63.9                                |

These improvements span mathematical reasoning, logical reasoning, coding, and comprehensive evaluation, which is consistent with intended role of recycled hiddens as an additional latent reasoning during generation.

Prefill recycle step $k = 1$ also retains a clear aggregate advantage while matching the baseline's single-pass prefill computation. It improves Avg. score from 62.9 to 65.3 (+2.4), with notable gains on HARP (+7.3), ARC-AGI (+6.8), LiveCodeBench (+6.1), SuperGPQA (+5.6), and AIME24 (+4.0). The $k = 2$-trained/$k = 1$-inference reference remains above the baseline in aggregate, but its smaller gain indicates that the strongest SFT performance is obtained when prefill recycle depth used at inference is aligned with $k = 2$ training path.

Overall, the SFT results show that Recycler improves broad aggregate performance after SFT, delivers its largest gains on reasoning-intensive tasks, and preserves a meaningful advantage even with prefill recycle step $k = 1$. Prefill recycle step $k = 2$ achieves the best overall quality, whereas $k = 1$ offers the more compute-efficient operating point for single-pass prefill.

## 3.3 Reinforcement Learning

### 3.3.1 Algorithm and Setup

We train the baseline and two Recycler variants with GSPO [65] on an internal math RL corpus in long-CoT format, using a rule-based verifier as the sole reward signal. Each run starts from its corresponding SFT checkpoint, with the Recycler variants retaining their SFT recycle steps of $k = 1$ and $k = 2$ throughout RL. All runs use the same per-step prompt and sampling budgets and are evaluated at mid-training checkpoints, as training was stopped manually before convergence.

### 3.3.2 Normal RL Results

We compare the three training routes introduced above: the non-Recycler baseline and the Recycler routes trained with recycle step $k = 1$ and $k = 2$. Since the recycle step is also an inference-time choice, we additionally evaluate the $k = 2$ route at $k = 1$, yielding four evaluation configurations in total. Table 2 reports these four alongside the SFT starting points of the three routes, which serve as the reference for the stage-wise comparison below. The recycle step is specific to Recycler and is therefore undefined for the baseline, whose inference cost matches the $k = 1$ configurations.

**Table 2** SFT and RL results on Open Benchmark. The first three rows are the SFT starting points of the three training routes; the remaining rows are their RL endpoints. Training $k$ and inference $k$ denote the recycle step used during training and at inference, respectively; both are undefined for the baseline, whose inference cost matches the $k = 1$ configurations. Bold marks the best value among the RL rows.

| System         | Training $k$ | Inference $k$ | Total     | Compr.    | Reason.   | Math      | Code      | Instr.-Follow | Knowl.    | Multil.   |
| -------------- | ------------ | ------------- | --------- | --------- | --------- | --------- | --------- | ------------- | --------- | --------- |
| Baseline (SFT) | -            | -             | 58.67     | 61.00     | 58.04     | 77.80     | 39.97     | 63.70         | 35.50     | 78.67     |
| Recycler (SFT) | 1            | 1             | 60.23     | 62.81     | 62.09     | 80.80     | 39.87     | 61.85         | 37.17     | 80.67     |
| Recycler (SFT) | 2            | 2             | 62.46     | 64.18     | 65.31     | 80.67     | 41.24     | 72.97         | 37.83     | 82.00     |
| Baseline       | -            | -             | 60.06     | 62.06     | 62.26     | 83.80     | 38.28     | 63.71         | 34.00     | 80.67     |
| Recycler       | 1            | 1             | 61.66     | 64.89     | 65.00     | 82.80     | 40.22     | 63.77         | 35.50     | 81.33     |
| Recycler       | 2            | 1             | 61.42     | 63.84     | 64.28     | 84.27     | 42.54     | 62.46         | 34.67     | 78.33     |
| Recycler       | 2            | 2             | **64.38** | **66.12** | **66.98** | **84.80** | **43.33** | **74.18**     | **37.17** | **83.67** |

As shown in Table 2, every Recycler configuration ends above the baseline after RL, whether the recycle step is varied at training time or at inference time. Trained and evaluated at $k = 2$, Recycler reaches 64.38 versus 60.06 for the baseline, a gain of 4.32 points, and leads on all seven ability groups. The advantage does not depend on retaining the second prompt forward: training and evaluating at $k = 1$ reaches 61.66 (+1.60), and the $k = 2$-trained route evaluated at $k = 1$ reaches 61.42 (+1.36). Both $k = 1$ configurations match the baseline's prefill and decode cost, so the improvement at these two operating points comes at no additional inference cost. The two $k = 1$ results are close in aggregate but reach it differently—the $k = 1$-trained model is stronger on comprehensive evaluation (+2.83 over baseline) and instruction following, whereas the $k = 2$-trained model evaluated at $k = 1$ is stronger on code (+4.26) and weaker on instruction following (−1.26) and multilingual evaluation (−2.33). The aggregate gains should therefore not be read as uniform improvement across every ability group.

**RL widens Recycler's advantage rather than merely preserving it.** Comparing each route with its own SFT starting point separates what RL adds from what Recycler already brought into RL. The baseline improves by 1.39 points from SFT to RL, and the $k = 1$ route by 1.43; the $k = 2$ route improves by 1.92 points when evaluated at $k = 2$ and by 1.76 when evaluated at $k = 1$. Consequently, the $k = 2$ route's margin over the baseline widens across the stage, from 3.79 points at SFT to 4.31 after RL, whereas the $k = 1$ route's margin is essentially unchanged ($1.56 \rightarrow 1.59$). The larger stage-wise increment of the $k = 2$ route is consistent with recycled hidden states supplying additional latent computation that RL can exploit, suggesting a higher performance ceiling than baseline under the same post-training recipe. Because the SFT/RL points are paired by route and three routes differ in code revision and rollout schedule, these increments support a descriptive comparison of complete training routes rather than an equal-compute claim about RL stage in isolation.

## 4 Related Work

We discuss related work along two lines. From a decoding perspective, Recycler feeds output hidden states forward to the next token, superficially resembling latent chain-of-thought (CoT) methods—though this is not its motivation. From a training perspective, Recycler also relates to models with recurrent structures.

### 4.1 Latent Chain-of-Thought

Chain-of-thought techniques let LLMs reason step by step through prompting [35, 55], fine-tuning [62], or reinforcement learning [52, 63]. Subsequent work pushes models to think longer before answering. Long-CoT models such as DeepSeek-R1 [24] realize test-time scaling and deliver substantial gains on mathematics and reasoning, adopting a reflective pattern that verifies candidate answers before responding.

Beyond natural-language CoT, several methods generate CoT in latent space. Goyal et al. [23] insert learnable `<pause>` tokens, while Su et al. [48] use an auxiliary codebook to interleave latent tokens with text. Most closely related is the autoregressive generation of continuous CoT tokens, where training is far less straightforward than inference. Xu et al. [57, 58] inject continuous tokens from an assistant model as latent CoTs. Hao et al. [25] propose COCONUT, which skips unembedding and feeds the output hidden state directly as the next token's input, adopting iCoT's progressive learning [12] to make training tractable. Tan et al. [50] build ground-truth continuous tokens by merging discrete CoT tokens and train with next-token prediction (NTP), whereas Shen et al. [47] derive them through self-distillation. Departing from COCONUT's direct reuse, Zhang et al. [64] reconstruct the hidden state from the previous token's NTP probabilities, replacing one-hot lookup with a weighted summation and bridging the input–output embedding gap under untied embeddings without training.

Recycler does not replace discrete CoT tokens with continuous counterparts; it lets the two coexist. Hidden states encoding rich context are preserved, while decoding still operates on discrete semantic tokens. Recycler therefore needs no explicit objective on output hidden states and trains with standard NTP alone.

### 4.2 Recursive Transformers

Recursive Transformers reuse the same module multiple times per prediction, adding computation but not parameters. Some loop-based methods call this latent thinking, but the computation is uniform across tokens, unlike the extensive pre-response reasoning of latent CoT.

Universal Transformer [11] first shows that repeating a single block can match a deeper model with unshared layers. Loop Transformers are formulated as programmable computers [21] and prove effective for iterative algorithms [59] and length generalization [14], and are now viewed as another dimension of test-time scaling [17, 46]. A second line routes over looped layers: Mixture-of-Recursions [2] assigns per-token recursion depths through a trainable router, and related schemes follow [8, 68]. From the recursive-Transformer perspective, Recycler reuses no intra-token hidden states, recycling only those of the previous token and thus incurring $1\times$ decoding overhead. More importantly, whereas recursive models typically supervise only the final forward pass, Recycler retains NTP at every step and benefits from future-token prediction signals.

Concurrent work likewise recycles hidden states. For example, Latent Recurrent Transformer [31] injects an intermediate source-layer state into the current token's layers via KV projection and residual injection, while Full-Bandwidth Transformer [53] fuses the previous top-layer state with the next-token embedding through a gated linear unit. Recycler is closer to the latter in injection site, but uses a simpler additive adapter with a shallow MTP-style multi-pass approximation that relies on decode-time extrapolation.

## 5 Conclusion

We introduced **Recycler**, a simple architecture that augments each token embedding with a lightweight projection of the preceding token's top-layer hidden state. This continuous cross-token pathway preserves information that would otherwise be discarded through discrete token sampling, while adding negligible overhead to autoregressive decoding. To train the resulting recurrent dependency efficiently, we develop a parallel multi-pass approximation that generalizes from a shallow recycle depth during training to long autoregressive decoding at inference. Across pre-training, continued pre-training, SFT, and RL, Recycler consistently improves model quality, with especially strong gains in reasoning and mathematics, and approaches the performance of a $2.4\times$ larger baseline at the decoding cost of the smaller model. Introducing Recycler during continued pre-training offers the best balance between training cost and downstream quality.

Together, these results establish hidden-state recycling as a practical axis for scaling latent computation across decoding steps, complementary to scaling model size or training data.
