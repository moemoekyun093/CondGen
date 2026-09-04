# Modular baseline commands

All paths below are runtime parameters. Replace `shoppers` with any dataset
whose `data/<name>/train.csv`, `test.csv`, and `info.json` exist.
The submission scripts export the checkout as `TABDIFF_PROJECT_ROOT`, because
Slurm runs worker copies from its spool directory rather than the source path.

## Script organization

These are the canonical user-facing entrypoints for query-conditioned work:

| Task | Entrypoint |
| --- | --- |
| Train DiffPuter or GReaT | `submit_native_query_baseline_train.sh` |
| Train a Doob query guide | `doob_query_train.sh` |
| Sample any combination of methods | `submit_query_suite_sampling.sh` |
| Evaluate existing samples only | `submit_query_suite_evaluation.sh` |

The two query-suite submission scripts accept paths at runtime, support any
dataset with TabDiff metadata, and accept repeated `--method` arguments. The
sampling entrypoint bundles query/seed pairs into a small number of long Slurm
jobs and reuses existing CSVs. The evaluation entrypoint never samples.

The following files are internal workers and normally should not be submitted
directly: `query_suite_sample_bundle.sh`, `query_suite_evaluate.sh`, and
`query_suite_alpha_evaluate.sh`.

The old `submit_query_split_comparison.sh`,
`submit_doob_harpoon_query_suite.sh`, and
`doob_harpoon_query_suite_evaluate.sh` names remain as deprecated compatibility
wrappers. They now forward to the canonical pipeline instead of maintaining a
second implementation. Fixed-query, clipping, architecture-ablation, and
historical TabDiff scripts are retained because their names encode published
experiment configurations; they are not general-purpose entrypoints.

## Train

```bash
bash submit_native_query_baseline_train.sh \
  --method diffputer --dataname shoppers \
  --output-dir baselines/checkpoints/shoppers/diffputer

bash submit_native_query_baseline_train.sh \
  --method great --dataname shoppers \
  --output-dir baselines/checkpoints/shoppers/great
```

Use `--train-data`, `--test-data`, and `--info-file` to bypass the conventional
dataset directory. Logs are in `logs/baselines/`. The scripts default to the
existing `tabdiff` Python for DiffPuter and dedicated `great` environment for GReaT; override with
`DIFFPUTER_PYTHON=/path/to/python` or `GREAT_PYTHON=/path/to/python` if the
baseline dependencies live in a dedicated environment. `alpha` remains used
only by the separate SynthCity metric job.

## Sample held-out queries (five seeds)

```bash
bash submit_query_suite_sampling.sh \
  --dataname shoppers \
  --query-dir data90/shoppers/queries \
  --query-split-manifest data90/shoppers/query_splits/sampled_arity_stratified_80_20_seed42.json \
  --query-split test \
  --num-seeds 5 \
  --sample-root conditional_samples/shoppers/native_baselines_test \
  --evaluation-output evaluations/shoppers/native_baselines_test \
  --method diffputer=diffputer:baselines/checkpoints/shoppers/diffputer \
  --method great=great:baselines/checkpoints/shoppers/great
```

With `--evaluation-output`, that one command submits bundled sampling and then
the complete evaluation with the correct Slurm dependency. Omit it when you
want sampling only.

The same command can include existing methods:

```text
--method doob=doob:tabdiff/ckpt/shoppers/.../guide_directory
--method harpoon=harpoon:/path/to/diffputer_selfmade.pt
--method harpoon_style_eta1=harpoon_style:1.0
```

For Doob, also pass `--base-checkpoint FILE`. Sampling jobs are bundled and
skip every CSV that already exists.

Use `--evaluation-method LABEL=SAMPLE_DIRECTORY` to add an existing method to
the downstream plots without scheduling any sampling for it.
Use `--evaluation-dependency JOB[:JOB...]` when those existing samples are
still being produced; unlike `--dependency`, it does not delay new sampling.

GReaT uses a total prompt-plus-generation limit of 512 tokens by default. This
avoids the upstream 0.0.9 default of 100 tokens being shorter than a
conditioned row prompt. Override it with `--great-max-length N` when needed.

## Evaluate existing samples only

```bash
bash submit_query_suite_evaluation.sh \
  --dataname shoppers \
  --query-dir data90/shoppers/queries \
  --query-split-manifest data90/shoppers/query_splits/sampled_arity_stratified_80_20_seed42.json \
  --query-split test \
  --num-seeds 5 \
  --output-dir evaluations/shoppers/native_baselines_test \
  --method doob=conditional_samples/shoppers/native_baselines_test/doob \
  --method harpoon=conditional_samples/shoppers/native_baselines_test/harpoon \
  --method diffputer=conditional_samples/shoppers/native_baselines_test/diffputer \
  --method great=conditional_samples/shoppers/native_baselines_test/great
```

Normal Shape/Trend/C2ST/XGBoost evaluation uses `relgdiff`; the separate Alpha
Precision/Beta Recall job uses `alpha`. Add `--skip-synthcity` to omit only the
latter. The evaluation command accepts sample directories and cannot sample.

The normal log is `evaluations/slurm/query_suite_eval_<JOB_ID>.out`; the
SynthCity log is `evaluations/slurm/query_alpha_eval_<JOB_ID>.out`. All CSVs
and plots go under the directory supplied via `--output-dir`.
