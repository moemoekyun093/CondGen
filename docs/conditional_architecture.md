# Conditional generation architecture and baseline protocol

## Frozen TabDiff FT-Transformer

The unconditional model is a diffusion denoiser. At a noise level/time `t`, it
receives the complete noisy numerical and categorical table row and predicts:

- one denoised value per numerical column; and
- unnormalized class logits for every categorical column.

Its FT-Transformer has three stages.

1. **Feature tokenization.** Every table column becomes one token. A numerical
   value uses a learned, column-specific periodic embedding
   `sin(2 pi f_j x_j), cos(2 pi f_j x_j)`, a learned projection, ReLU, and a
   column bias. A categorical state is one-hot and selects/sums the learned
   category lookup vectors for that column, followed by its column bias.
2. **Contextual denoising.** A sinusoidal embedding of `t` is projected and
   injected at every transformer layer. Each layer applies pre-normalized
   multi-head self-attention across *all column tokens*, a residual connection,
   then a token-wise GELU feed-forward network and another residual. Attention
   is the mechanism that lets, for example, the `ExitRates` prediction depend
   on every other noisy column in the row.
3. **Reconstruction.** A per-column reconstructor maps contextual tokens back
   to numerical predictions and categorical logits. There is no sparse feature
   mask and no flatten-to-MLP bottleneck in this active backbone.

The base denoiser is frozen while a Doob guide is trained.

## Structured Doob conditional model

There are two lightweight guide networks with the same token architecture but
separate parameters:

- the numerical guide predicts the whole-row vector
  `grad_(x_num) log h(t, x, query)`;
- the categorical guide predicts the scalar `log h(t, x, query)`. During
  categorical sampling it is evaluated at candidate child states and reweights
  the frozen base generator using `h(child) / h(current)` (implemented stably
  as log-h differences / candidate logits).

Both guides see the entire noisy numerical and categorical state, time, and
the entire query. They do not use one network per column.

For the current per-token center/log-width conditioner:

1. The frozen base tokenizer converts the noisy row into one token per column.
   The same numerical EDM input scaling used by the base model is applied
   before this frozen tokenization.
2. Each base token is projected to the smaller guide width. Time is embedded,
   concatenated to every projected state token, and fused by `Linear + SiLU`.
3. An active numerical interval `[l_j, u_j]` is represented by
   `c_j=(l_j+u_j)/2` and `log(max(u_j-l_j, eps))`. Two ordinary scalar MLPs
   embed these clean query values. The raw center, raw log-width, both
   embeddings, and the matching noisy state token are concatenated, then
   reduced to one guide token by `Linear + SiLU`.
4. A categorical allowed set is represented using the frozen base tokenizer's
   real-category lookup vectors: allowed vectors are ReLUed and summed. The
   MASK lookup and tokenizer column bias are excluded. This set vector is
   concatenated only with the matching categorical state token and fused by
   `Linear + SiLU`.
5. Dense FT blocks then allow every fused token to attend to every other fused
   token. Thus query information enters locally by column first, but the final
   correction is conditioned on the whole row and whole query.

The legacy active-flag, implicit-domain, endpoint, monotone, and alternating
constraint-token variants remain selectable through the saved guide config;
the sampler reconstructs the architecture from that checkpoint.

## DiffPuter and GReaT comparison protocol

HARPOON's released general-constraint comparison calls its one-hot DDPM with
RePaint-style observed-cell injection `DiffPuter_Remastered`. This is not the
original DiffPuter EDM/EM implementation. The modular CLI therefore records
the method as `diffputer_harpoon_repaint` in checkpoint/sample metadata rather
than silently conflating the two algorithms.

- **Released DiffPuter/RePaint baseline:** an MLP noise predictor operates on
  standardized numerical columns plus one-hot categorical columns for 200
  diffusion steps. At every reverse step, observed categorical cells are
  replaced with their correctly forward-noised known values.
- **GReaT:** a DistilGPT-2 autoregressive language model serializes table rows
  and is fine-tuned on the training table. Sampling uses its native imputation
  API for observed cells.

Neither released baseline defines a differentiable operator for a numerical
interval. To remain faithful to HARPOON's comparison, numerical `between`
predicates are left missing/unconditional and their violations are measured
after generation. A categorical `in` set must be converted to the exact-cell
interface: for every generated row, the adapter draws one allowed category
from the empirical training marginal restricted to the allowed set. This
extension is deterministic under the requested sampling seed and is written
to each sample's metadata JSON.

## Modular interfaces

Training consumes either `--dataname NAME` (the `data/NAME` convention) or
explicit `--train-data`, `--test-data`, and `--info-file` paths. Sampling
consumes model directories and query JSON files. Evaluation consumes only
sample directories, so it never retrains or resamples.

The first sampling seed retains the historical direct layout
`METHOD/QUERY.csv`; additional seeds use
`METHOD/seed_SEED_BASE/QUERY.csv`. The evaluator averages per-query metrics
over every supplied seed before aggregating by selectivity, arity, or mean
transformed interval width.
