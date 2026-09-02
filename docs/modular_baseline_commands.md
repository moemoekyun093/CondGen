# Modular baseline commands

All paths below are runtime parameters. Replace `shoppers` with any dataset
whose `data/<name>/train.csv`, `test.csv`, and `info.json` exist.
The submission scripts export the checkout as `TABDIFF_PROJECT_ROOT`, because
Slurm runs worker copies from its spool directory rather than the source path.

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
existing `tabdiff` Python for DiffPuter and `relgdiff` for GReaT; override with
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
```

For Doob, also pass `--base-checkpoint FILE`. Sampling jobs are bundled and
skip every CSV that already exists.

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
