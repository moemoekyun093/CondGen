# TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation

<p align="center">
  <a href="https://github.com/MinkaiXu/TabDiff/blob/main/LICENSE">
    <img alt="MIT License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a>
  <a href="https://openreview.net/forum?id=swvURjrt8z">
    <img alt="Openreview" src="https://img.shields.io/badge/review-OpenReview-blue">
  </a>
  <a href="https://arxiv.org/abs/2410.20626">
    <img alt="Paper URL" src="https://img.shields.io/badge/cs.LG-2410.20626-B31B1B.svg">
  </a>
</p>

<div align="center">
  <img src="images/tabdiff_demo.gif" alt="Model Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <p><em>Figure 1: Visualing the generative process of TabDiff. A high-quality version of this video can be found at <a href="images/tabdiff_demo.mp4" download>tabdiff_demo.mp4</a></em></p>
</div>

This repository provides the official implementation of TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation (ICLR 2025).

## Latest Update
- [2025.04]：The categorical-heavy dataset **[Diabetes](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)** evaluated in the paper has now been released!
- [2025.02]：Our code is finally released! We have released part of the tested datasets. The rest will be released soon!

## Introduction

<div align="center">
  <img src="images/tabdiff_flowchart.jpg" alt="Model Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <p><em>Figure 2: The high-level schema of TabDiff</a></em></p>
</div>
TabDiff is a unified diffusion framework designed to model all muti-modal distributions of tabular data in a single model. Its key innovations include:  

1) Framing the joint diffusion process in continuous time,
2) A feature-wised learnable diffusion process that offsets the heterogeneity across different feature distributions,
3) Classifier-free guidance conditional generation for missing column value imputation. 

The schema of TabDiff is presented in the figure above. For more details, please refer to [our paper](https://arxiv.org/abs/2410.20626).


## Environment Setup

Create the main environment with [tabdiff.yaml](tabdiff.yaml). This environment will be used for all tasks except for the evaluation of additional data fidelity metrics (i.e., $\alpha$-precision and $\beta$-recall scores)

```
conda env create -f tabdiff.yaml
```

Create another environment with [synthcity.yaml](synthcity.yaml) to evaluate additional data fidelity metrics

```
conda env create -f synthcity.yaml
```

## Datasets Preparation

### Using the datasets experimented in the paper

Download raw datasets:

```
python download_dataset.py
```

Process datasets:

```
python process_dataset.py
```

### Using your own dataset

First, create a directory for your dataset in [./data](./data):
```
cd data
mkdir <NAME_OF_YOUR_DATASET>
```

Compile your raw tabular data in .csv format. **The first row should be the header** indicating the name of each column, and the remaining rows are records. After finishing these steps, place you data's csv file in the directory you just created and name it as <NAME_OF_YOUR_DATASET>.csv. 

Then, create <NAME_OF_YOUR_DATASET>.json in [./data/Info](./data/Info). Write this file with the metadata of your dataset, covering the following information:
```
{
    "name": "<NAME_OF_YOUR_DATASET>",
    "task_type": "[NAME_OF_TASK]", # binclass or regression
    "header": "infer",
    "column_names": null,
    "num_col_idx": [LIST],  # list of indices of numerical columns
    "cat_col_idx": [LIST],  # list of indices of categorical columns
    "target_col_idx": [list], # list of indices of the target columns (for MLE)
    "file_type": "csv",
    "data_path": "data/<NAME_OF_YOUR_DATASET>/<NAME_OF_YOUR_DATASET>.csv"
    "test_path": null,
}
```

### Important Notes When Creating the Info File
- The MLE evaluation and the imputation task (see later sections for details) assume that one column of your data is the regression or classification target. To enable these tasks, you will need to specify `target_col_idx`. If you don't need to evalute MLE, you can comment out the following line: https://github.com/MinkaiXu/TabDiff/blob/0c4fc3bbfa19046d36c5dce64628df52d5c73d15/tabdiff/main.py#L152
- The fields `target_col_idx`, `num_col_idx` and `cat_col_idx` must be multually exclusive—no column should appear in more than one of these lists. 
- Set the task_type to "regression" if the target column is numerical, or "binclass" if it is categorical.

Finally, run the following command to process your dataset:
```
python process_dataset.py --dataname <NAME_OF_YOUR_DATASET>
```

## Training TabDiff

To train an unconditional TabDiff model across the entire table, run

```
python main.py --dataname <NAME_OF_DATASET> --mode train
```

Current Options of ```<NAME_OF_DATASET>``` are: adult, default, shoppers, magic, beijing, news

Wanb logging is enabled by default. To disable it and log locally, add the ```--no_wandb``` flag.

To disable the learnable noise schedules, add the ```--non_learnable_schedule```. Please note that in order for the code to test/sample from such model properly, you need to add this flag for all commands below.

To specify your own experiment name, which will be used for logging and saving files, add ```--exp_name <your experiment name>```. This flag overwrites the default experiment name (learnable_schedule/non_learnable_schedule), so, similar to ```--non_learnable_schedule```, once added to training, you need to add it to all following commands as well.

## Sampling and Evaluating TabDiff (Density, MLE, C2ST)

To sample synthetic tables from trained TabDiff models and evaluate them, run
```
python main.py --dataname <NAME_OF_DATASET> --mode test --report --no_wandb
```

This will sample 20 synthetic tables randomly. Meanwhile, it will evaluate the density, mle, and c2st scores for each sample and report their average and standard deviation. The results will be printed out in the terminal, and the samples and detailed evaluation results will be placed in ./eval/report_runs/<EXP_NAME>/<NAME_OF_DATASET>/.

## Evaluating on Additional Fidelity Metrics ($\alpha$-precision and $\beta$-recall scores)
To evaluate TabDiff on the additional fidelity metrics ($\alpha$-precision and $\beta$-recall scores), you need to first make sure that you have already generated some samples by the previous commands. Then, you need to switch to the `synthcity` environment (as the synthcity packet used to compute those metrics conflicts with the main environment), by running
```
conda activate synthcity
```
Then, evaluate the metrics by running
```
python eval/eval_quality.py --dataname <NAME_OF_DATASET>
```

Similarly, the results will be printed out in the terminal and added to ./eval/report_runs/<EXP_NAME>/<NAME_OF_DATASET>/

## Evaluating Data Privacy (DCR score)
To evalute the privacy metric DCR score, you first need to retrain all the models, as the metric requires an equal split between the training and testing data (our initial splits employ a 90/10 ratio). To retrain with an equal split, run the training command but append `_dcr` to ```<NAME_OF_DATASET>```
```
python main.py --dataname <NAME_OF_DATASET>_dcr --mode train
```

Then, test the models on DCR with the same `_dcr` suffix
```
python main.py --dataname <NAME_OF_DATASET>_dcr --mode test --report --no_wandb
```



## Missing Value Imputation with Classifier-free Guidance (CFG)
Our current experiments only include imputing the target column. However, our implementation, located at ```sample_impute()``` in [unified_ctime_diffusion.py](./tabdiff/models/unified_ctime_diffusion.py), should support imputing multiple columns with different data types.

### Training Guidance Model
In order to enable classifier-free guidance (CFG), you need to first train an unconditional guidance model on the target column by running the training command with the `--y_only` flag
```
python main.py --dataname <NAME_OF_DATASET> --mode train --y_only
```

### Sampling Imputed Tables
With the trained guidance model, you can then impute the missing target column by running the testing command with the `--impute` flag
```
python main.py --dataname <NAME_OF_DATASET> --mode test --impute --no_wandb
```
This will, by default, randomly produce 50 imputed tables and save them to ./impute/<NAME_OF_DATASET>/<EXP_NAME>.

### Evaluating Imputation
You can then evaluate the imputation quality by running
```
python eval_impute.py --dataname <NAME_OF_DATASET>
```

## Fixed-query numerical generation with a Doob h-transform

This experimental path learns one broad numerical query after the unconditional
TabDiff model has been trained.  The base model and noise schedules remain frozen.
For a terminal event `B`, the auxiliary FT-periodic network learns

```
delta_D(x_t, t) = sigma(t)^2 * grad_x log P(X_0 in B | X_t=x_t).
```

The stable training target derived from the score-matching objective is
`x_0 - D_base(x_t, t)`.  At sampling time the conditional denoiser is
`D_B = D_base + guidance_strength * delta_D`.

The first experiment uses one deterministic fixed box over every normalized
numerical column.  A preprocessing job chooses a shared symmetric marginal
quantile level so the joint box contains at least 30% of transformed training
rows, then saves the normalized and raw-space bounds to JSON.  The query is not
an input to the guide: the guide is parameterized only by the complete noisy row
and time. Categorical values are not part of the constraint, but the guide also
learns the scalar value `h(t,x) = P(X_0 in B | X_t=x)`. During sampling,
categorical reverse transitions are reweighted by the mixed Doob ratio
`h(t,u,c_child) / h(t,u,c_current)` while the original TabDiff logits and mask
schedule remain frozen.

For optional categorical equality constraints, the sampler also implements the
Section 4 ordering construction. If categorical columns `C` are fixed, it draws
each trajectory's reverse start time from the discretized posterior
`q_C(t)` for paths on which every column in `C` is revealed before every free
categorical column. It starts with `C` fixed and all other categorical entries
masked. The guide is still trained on the original forward-corruption law
`q(x_t | x_0)`: `q_C(t)` changes sampling initialization and is not a new
guide-training distribution. With the present numerical-only query there are no
fixed categorical entries, so `q_C` is undefined and the sampler correctly
falls back to the original `t=1` start; categorical `h(child)/h(current)`
reweighting remains active.

Generate and inspect the deterministic intervals first:

```bash
mkdir -p logs
INTERVAL_JOB=$(sbatch --parsable doob_h_intervals.sh)
echo "interval job: ${INTERVAL_JOB}"
```

After that job finishes, inspect and version the fixed query:

```bash
cat constraints/shoppers/fixed_numerical_intervals.json
```

Then submit the two-task guide-training array. For the default Shoppers
experiment, task 0 uses `ft_periodic_seed0`; task 1 uses `original_seed0`. Both
use the same fixed query and train separate guides:

```bash
sbatch doob_h_train.sh
```

After training succeeds, submit conditional generation:

```bash
sbatch doob_h_sample.sh
```

To add categorical equalities, pass comma-separated model-space
`COLUMN=CLASS` indices. For example, this fixes categorical model column 0 to
class 1 and column 3 to class 2, then uses the Section 4 posterior start:

```bash
export FIXED_CATEGORICAL="0=1,3=2"
sbatch --export=ALL,DATANAME=adult,SAMPLE_SUFFIX=_mixed_cat doob_h_sample.sh
```

These are encoded model indices, not raw CSV labels. The job prints the sampled
start-time mean/min/max. Use a distinct `SAMPLE_SUFFIX` for each categorical
query so results are not overwritten.

The mixed-guidance sampling array writes model-specific outputs to
`conditional_samples/shoppers/ft_periodic_seed0_mixed.csv` and
`conditional_samples/shoppers/original_seed0_mixed.csv`, with a matching
`.constraints.json` report beside each CSV.

To submit the complete dependency chain:

```bash
INTERVAL_JOB=$(sbatch --parsable doob_h_intervals.sh)
TRAIN_JOB=$(sbatch --parsable --dependency="afterok:${INTERVAL_JOB}" doob_h_train.sh)
sbatch --dependency="afterok:${TRAIN_JOB}" doob_h_sample.sh
```

Guide training follows the base TabDiff trainer's data-use and checkpointing
conventions: all rows train the scalar `h` value while rows satisfying the fixed
query train the numerical score correction (there is no validation split).
Training runs for 8000 full epochs by default, and the learning
rate uses reduce-on-plateau on the combined numerical-correction MSE and scalar
`h` binary cross-entropy, and an EMA guide with decay
0.997 is updated after every epoch.  Best raw and EMA checkpoint selection starts
after epoch 4000, and `best_guide.pt` is the selected EMA guide used for sampling.
Training parameters can be overridden with SLURM exports, for example
`sbatch --export=ALL,EPOCHS=5000,BATCH_SIZE=2048 doob_h_train.sh`. Guidance
strength is fixed to 1.0 for this experiment.

The dataset and checkpoint-directory names can also be overridden without
editing the scripts, for example
`sbatch --export=ALL,DATANAME=shoppers,FT_MODEL=ft_periodic_small_d64_L4_seed0 doob_h_train.sh`.

The sampler uses guidance strength 1.0 and writes both the generated CSV and a
`.constraints.json` report with raw and normalized-space joint and per-column
hit rates. `joint_hit_rate` is evaluated on the final inverse-transformed table
in raw units; the normalized rate is retained as a diagnostic because quantile
normalization is not one-to-one for repeated values. A 100% raw joint hit rate
means every generated numerical row satisfied every saved user-facing interval.
The default per-coordinate correction clamp is 5 normalized units and can be
changed with `--max-correction`.

Existing conditional CSVs can be diagnosed and evaluated without sampling
again. The CPU-only evaluation array recomputes raw-space constraint hit rates,
then uses TabDiff's SDMetrics Shape and Trend implementation twice: primarily
against real rows satisfying the same raw constraint, and secondarily against
the full real training table.

```bash
sbatch doob_h_evaluate.sh
```

Results are written under
`conditional_samples/shoppers/<MODEL_NAME>_mixed_evaluation/density_results.json`;
the detailed per-column Shape and column-pair Trend tables are saved beside it.
The same evaluation also saves Pearson correlation matrices for the full real
table, constrained real subset, and generated table, plus summaries of
constrained-vs-total and generated-vs-constrained correlation changes.

## License

This work is licensed undeer the MIT License.

## Acknowledgement
This repo is built upon the previous work TabSyn's [[codebase]](https://github.com/amazon-science/tabsyn). Many thanks to Hengrui!

## Citation
Please consider citing our work if you find it helpful in your research!
```
@inproceedings{
shi2025tabdiff,
title={TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation},
author={Juntong Shi and Minkai Xu and Harper Hua and Hengrui Zhang and Stefano Ermon and Jure Leskovec},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=swvURjrt8z}
}
```
## Contact
If you encounter any problem, please file an issue on this GitHub repo.

If you have any question regarding the paper, please contact Minkai at [minkai@stanford.edu](minkai@stanford.edu) or Juntong at [shisteve@usc.edu](shisteve@usc.edu).
