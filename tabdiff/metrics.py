from copy import deepcopy
import numpy as np
import torch
import pandas as pd
# Metrics
from eval.visualize_density import plot_density
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
from sdmetrics.single_table import LogisticDetection
from sklearn.preprocessing import OneHotEncoder
from sdmetrics.single_table.detection.sklearn import ScikitLearnClassifierDetectionMetric
from xgboost import XGBClassifier

from tqdm import tqdm


class TabMetrics(object):
    def __init__(
            self, real_data_path, test_data_path, val_data_path, info, device,
            metric_list, include_density_diagnostic=True
        ) -> None:
        self.real_data_path = real_data_path
        self.test_data_path = test_data_path
        self.val_data_path = val_data_path
        self.info = info
        self.device = device
        self.real_data_size = len(pd.read_csv(real_data_path))
        self.metric_list = metric_list
        self.include_density_diagnostic = include_density_diagnostic

    def evaluate(self, syn_data):
        metrics, extras = {}, {}
        syn_data_cp = deepcopy(syn_data)
        for metric in self.metric_list:
            func = eval(f"self.evaluate_{metric}")
            print(f"Evaluating {metric}")
            out_metrics, out_extras = func(syn_data_cp)
            metrics.update(out_metrics)
            extras.update(out_extras)
        return metrics, extras
    
    def evaluate_density(self, syn_data):
        real_data = pd.read_csv(self.real_data_path)
        real_data.columns = range(len(real_data.columns))
        syn_data.columns = range(len(syn_data.columns))
        

        info = deepcopy(self.info)
        
        y_only = len(syn_data.columns)==1
        if y_only:
            target_col_idx = info['target_col_idx'][0]
            syn_data = self.complete_y_only_data(syn_data, real_data, target_col_idx)

        metadata = info['metadata']
        metadata['columns'] = {int(key): value for key, value in metadata['columns'].items()} # ensure that keys are all integers?

        new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

        qual_report = QualityReport()
        qual_report.generate(new_real_data, new_syn_data, metadata)

        quality =  qual_report.get_properties()
        if self.include_density_diagnostic:
            diag_report = DiagnosticReport()
            diag_report.generate(new_real_data, new_syn_data, metadata)

        Shape = quality['Score'][0]
        Trend = quality['Score'][1]

        Overall = (Shape + Trend) / 2

        shape_details = qual_report.get_details(property_name='Column Shapes')
        trend_details = qual_report.get_details(property_name='Column Pair Trends')

        if y_only:
            Shape = shape_details['Score'].min()
        out_metrics = {
            "density/Shape": Shape,
            "density/Trend": Trend,
            "density/Overall": Overall,
        }
        out_extras = {
            "shapes": shape_details,
            "trends": trend_details
        }
        return out_metrics, out_extras
    
    def evaluate_mle(self, syn_data):
        # MLE is optional. Import its additional PRDC dependency only when the
        # caller explicitly requests the MLE metric; density/C2ST evaluations
        # should not require it merely to import TabMetrics.
        from eval.mle.mle import get_evaluator

        train = syn_data.to_numpy()
        test = pd.read_csv(self.test_data_path).to_numpy()
        val = pd.read_csv(self.val_data_path).to_numpy() if self.val_data_path else None
        
        info = deepcopy(self.info)

        task_type = info['task_type']

        evaluator = get_evaluator(task_type)

        if task_type == 'regression':
            best_r2_scores, best_rmse_scores = evaluator(train, test, info, val=val)
            
            overall_scores = {}
            for score_name in ['best_r2_scores', 'best_rmse_scores']:
                overall_scores[score_name] = {}
                
                scores = eval(score_name)
                for method in scores:
                    name = method['name']  
                    method.pop('name')
                    overall_scores[score_name][name] = method 

        else:
            best_f1_scores, best_weighted_scores, best_auroc_scores, best_acc_scores, best_avg_scores = evaluator(train, test, info, val=val)

            overall_scores = {}
            for score_name in ['best_f1_scores', 'best_weighted_scores', 'best_auroc_scores', 'best_acc_scores', 'best_avg_scores']:
                overall_scores[score_name] = {}
                
                scores = eval(score_name)
                for method in scores:
                    name = method['name']  
                    method.pop('name')
                    overall_scores[score_name][name] = method
                    
        mle_score = overall_scores['best_rmse_scores']['XGBRegressor']['RMSE'] if task_type == 'regression' else overall_scores['best_auroc_scores']['XGBClassifier']['roc_auc']
        out_metrics = {
            "mle": mle_score,
        }
        out_extras = {
            "mle": overall_scores,
        }
        return out_metrics, out_extras
    
    def evaluate_c2st(self, syn_data):
        info = deepcopy(self.info)
        real_data = pd.read_csv(self.real_data_path)

        real_data.columns = range(len(real_data.columns))
        syn_data.columns = range(len(syn_data.columns))

        metadata = info['metadata']
        metadata['columns'] = {int(key): value for key, value in metadata['columns'].items()}

        new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

        score = LogisticDetection.compute(
            real_data=new_real_data,
            synthetic_data=new_syn_data,
            metadata=metadata
        )
        
        out_metrics = {
            "c2st": score,
        }
        out_extras = {}
        return out_metrics, out_extras

    def evaluate_dcr(self, syn_data):
        info = deepcopy(self.info)
        real_data = pd.read_csv(self.real_data_path)
        test_data = pd.read_csv(self.test_data_path)
        
        num_col_idx = info['num_col_idx']
        cat_col_idx = info['cat_col_idx']
        target_col_idx = info['target_col_idx']

        task_type = info['task_type']
        if task_type == 'regression':
            num_col_idx += target_col_idx
        else:
            cat_col_idx += target_col_idx

        num_ranges = []

        real_data.columns = list(np.arange(len(real_data.columns)))
        syn_data.columns = list(np.arange(len(real_data.columns)))
        test_data.columns = list(np.arange(len(real_data.columns)))
        for i in num_col_idx:
            num_ranges.append(real_data[i].max() - real_data[i].min()) 
        
        num_ranges = np.array(num_ranges)


        num_real_data = real_data[num_col_idx]
        cat_real_data = real_data[cat_col_idx]
        num_syn_data = syn_data[num_col_idx]
        cat_syn_data = syn_data[cat_col_idx]
        num_test_data = test_data[num_col_idx]
        cat_test_data = test_data[cat_col_idx]

        num_real_data_np = num_real_data.to_numpy()
        cat_real_data_np = cat_real_data.to_numpy().astype('str')
        num_syn_data_np = num_syn_data.to_numpy()
        cat_syn_data_np = cat_syn_data.to_numpy().astype('str')
        num_test_data_np = num_test_data.to_numpy()
        cat_test_data_np = cat_test_data.to_numpy().astype('str')

        encoder = OneHotEncoder()
        cat_complete_data_np = np.concatenate([cat_real_data_np, cat_test_data_np], axis=0)
        encoder.fit(cat_complete_data_np)
        # encoder.fit(cat_real_data_np)


        cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
        cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()
        cat_test_data_oh = encoder.transform(cat_test_data_np).toarray()

        num_real_data_np = num_real_data_np / num_ranges
        num_syn_data_np = num_syn_data_np / num_ranges
        num_test_data_np = num_test_data_np / num_ranges

        real_data_np = np.concatenate([num_real_data_np, cat_real_data_oh], axis=1)
        syn_data_np = np.concatenate([num_syn_data_np, cat_syn_data_oh], axis=1)
        test_data_np = np.concatenate([num_test_data_np, cat_test_data_oh], axis=1)

        device = self.device

        real_data_th = torch.tensor(real_data_np).to(device)
        syn_data_th = torch.tensor(syn_data_np).to(device)  
        test_data_th = torch.tensor(test_data_np).to(device)

        dcrs_real = []
        dcrs_test = []
        batch_size = 10000 // cat_real_data_oh.shape[1]   # This esitmation should make sure that dcr_real and dcr_test can be fit into 10GB GPU memory

        for i in tqdm(range((syn_data_th.shape[0] // batch_size) + 1)):
            if i != (syn_data_th.shape[0] // batch_size):
                batch_syn_data_th = syn_data_th[i*batch_size: (i+1) * batch_size]
            else:
                batch_syn_data_th = syn_data_th[i*batch_size:]
                
            dcr_real = (batch_syn_data_th[:, None] - real_data_th).abs().sum(dim = 2).min(dim = 1).values
            dcr_test = (batch_syn_data_th[:, None] - test_data_th).abs().sum(dim = 2).min(dim = 1).values
            dcrs_real.append(dcr_real)
            dcrs_test.append(dcr_test)
            
        dcrs_real = torch.cat(dcrs_real)
        dcrs_test = torch.cat(dcrs_test)
        
        score = (dcrs_real < dcrs_test).nonzero().shape[0] / dcrs_real.shape[0]
        
        out_metrics = {
            "dcr": score,
        }
        out_extras = {
            "dcr_real": dcrs_real.cpu().numpy(),
            "dcr_test": dcrs_test.cpu().numpy(),
        }
        return out_metrics, out_extras

    def evaluate_mi_l1(self, syn_data):
        """
        Mutual-information dependency fidelity. Bins all columns with shared edges
        (from real data), computes the pairwise MI matrix for real vs synthetic,
        and reports the L1 error. The weighted complement is the headline number:
        1.0 = dependencies perfectly preserved, lower = correlations damaged.
        This is the metric that detects whether sparse feature masking broke
        feature-feature dependency structure.
        """
        from sklearn.metrics import mutual_info_score

        real_data = pd.read_csv(self.real_data_path)
        real_data.columns = range(len(real_data.columns))
        syn_data = syn_data.copy()
        syn_data.columns = range(len(syn_data.columns))

        info = deepcopy(self.info)
        num_col_idx = deepcopy(info['num_col_idx'])
        cat_col_idx = deepcopy(info['cat_col_idx'])
        target_col_idx = deepcopy(info['target_col_idx'])
        if info['task_type'] == 'regression':
            num_col_idx = num_col_idx + target_col_idx
        else:
            cat_col_idx = cat_col_idx + target_col_idx

        n_bins = 10
        # Build shared bin edges from REAL data so real/syn are directly comparable
        bin_edges = {}
        for c in num_col_idx:
            bin_edges[c] = np.histogram_bin_edges(real_data[c].astype(float), bins=n_bins)

        def discretize(df):
            out = pd.DataFrame(index=df.index)
            for c in num_col_idx:
                out[c] = np.digitize(df[c].astype(float), bin_edges[c])
            for c in cat_col_idx:
                out[c] = df[c].astype(str)
            return out

        binned_real = discretize(real_data)
        binned_syn = discretize(syn_data)
        cols = list(binned_real.columns)
        n = len(cols)

        def mi_matrix(binned):
            M = np.zeros((n, n))
            for a in range(n):
                for b in range(a):  # lower triangle only
                    M[a, b] = mutual_info_score(binned[cols[a]], binned[cols[b]])
            return M

        mi_real = mi_matrix(binned_real)
        mi_syn = mi_matrix(binned_syn)

        tril = np.tril_indices(n, k=-1)
        err = np.abs(mi_real[tril] - mi_syn[tril])

        weights = mi_real[tril] + mi_syn[tril]
        weights_sum = weights.sum() + 1e-12

        mi_l1 = float(err.mean())
        mi_l1_weighted = float((weights * err).sum() / weights_sum)

        out_metrics = {
            "mi/l1": mi_l1,
            "mi/l1_weighted": mi_l1_weighted,
            "mi/l1_complement": 1.0 - mi_l1,
            "mi/l1_weighted_complement": 1.0 - mi_l1_weighted,  # <- headline: higher is better
        }
        out_extras = {}
        return out_metrics, out_extras

    def evaluate_quantile_dcr(self, syn_data):
        """
        Memorization-aware DCR. Unlike evaluate_dcr (which asks 'is syn closer to
        real-train than to test, on average'), this measures what FRACTION of synthetic
        rows sit closer to a training record than genuinely-held-out test data does
        (at the 2% and 5% quantiles of test-to-train distance). A high fraction =
        synthetic rows are hugging training data = memorization.
        """
        info = deepcopy(self.info)
        real_data = pd.read_csv(self.real_data_path)
        test_data = pd.read_csv(self.test_data_path)

        num_col_idx = deepcopy(info['num_col_idx'])
        cat_col_idx = deepcopy(info['cat_col_idx'])
        target_col_idx = deepcopy(info['target_col_idx'])
        if info['task_type'] == 'regression':
            num_col_idx = num_col_idx + target_col_idx
        else:
            cat_col_idx = cat_col_idx + target_col_idx

        real_data.columns = list(np.arange(len(real_data.columns)))
        syn_data = syn_data.copy()
        syn_data.columns = list(np.arange(len(real_data.columns)))
        test_data.columns = list(np.arange(len(real_data.columns)))

        # Numeric: standardize by real mean/std so all columns contribute comparably
        num_real = real_data[num_col_idx].to_numpy(dtype=float)
        num_syn = syn_data[num_col_idx].to_numpy(dtype=float)
        num_test = test_data[num_col_idx].to_numpy(dtype=float)
        if len(num_col_idx) > 0:
            mean = num_real.mean(axis=0)
            std = num_real.std(axis=0) + 1e-8
            num_real = (num_real - mean) / std
            num_syn = (num_syn - mean) / std
            num_test = (num_test - mean) / std

        # Categorical: one-hot on combined real+test vocabulary
        if len(cat_col_idx) > 0:
            cat_real = real_data[cat_col_idx].to_numpy().astype(str)
            cat_syn = syn_data[cat_col_idx].to_numpy().astype(str)
            cat_test = test_data[cat_col_idx].to_numpy().astype(str)
            encoder = OneHotEncoder(handle_unknown='ignore')
            encoder.fit(np.concatenate([cat_real, cat_test], axis=0))
            cat_real = encoder.transform(cat_real).toarray()
            cat_syn = encoder.transform(cat_syn).toarray()
            cat_test = encoder.transform(cat_test).toarray()
        else:
            cat_real = np.empty((len(num_real), 0))
            cat_syn = np.empty((len(num_syn), 0))
            cat_test = np.empty((len(num_test), 0))

        real_np = np.concatenate([num_real, cat_real], axis=1)
        syn_np = np.concatenate([num_syn, cat_syn], axis=1)
        test_np = np.concatenate([num_test, cat_test], axis=1)

        device = self.device
        real_th = torch.tensor(real_np, dtype=torch.float32).to(device)
        syn_th = torch.tensor(syn_np, dtype=torch.float32).to(device)
        test_th = torch.tensor(test_np, dtype=torch.float32).to(device)

        eff_feats = max(1, real_th.shape[1])
        batch_size = max(50, 10000 // eff_feats)

        def min_dcr(source_th, target_th):
            out = []
            for i in range((source_th.shape[0] // batch_size) + 1):
                start, end = i * batch_size, min((i + 1) * batch_size, source_th.shape[0])
                if start >= end:
                    continue
                b = source_th[start:end]
                d = (b[:, None, :] - target_th[None, :, :]).abs().sum(2).min(1).values
                out.append(d)
            return torch.cat(out)

        dcrs_syn = min_dcr(syn_th, real_th)     # syn -> nearest real-train
        dcrs_test = min_dcr(test_th, real_th)   # test -> nearest real-train (honest baseline)

        ps = [0.02, 0.05]
        out_metrics = {}
        for p in ps:
            thresh = torch.quantile(dcrs_test, p)
            frac = (dcrs_syn < thresh).float().mean().item() * 100
            out_metrics[f"quantile_dcr/frac_below_{p}"] = frac  # <- higher = more memorization
        out_metrics["quantile_dcr/syn_mean"] = float(dcrs_syn.mean().cpu())
        out_metrics["quantile_dcr/test_mean"] = float(dcrs_test.mean().cpu())

        # out_extras = {
        #     "quantile_dcr_syn": dcrs_syn.cpu().numpy().tolist(),
        #     "quantile_dcr_test": dcrs_test.cpu().numpy().tolist(),
        # }
        out_extras = {}
        return out_metrics, out_extras
    
    def evaluate_c2st_xgb(self, syn_data):
        info = deepcopy(self.info)
        real_data = pd.read_csv(self.real_data_path)

        real_data.columns = range(len(real_data.columns))
        syn_data.columns = range(len(syn_data.columns))

        metadata = info['metadata']
        metadata['columns'] = {int(key): value for key, value in metadata['columns'].items()}

        new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

        score = XGBoostDetection.compute(
            real_data=new_real_data,
            synthetic_data=new_syn_data,
            metadata=metadata
        )

        out_metrics = {
            "c2st_xgb": score,
            "c2st_xgb_auc": 1 - score / 2,   # convert detection score to an AUC-style number
        }
        out_extras = {}
        return out_metrics, out_extras  

    def evaluate_c2st_xgb_val(self, syn_data):
        """
        Selection-only metric: identical to evaluate_c2st_xgb but compares synthetic
        against the VALIDATION set instead of the training data. Used for hyperparameter
        selection so we never tune against the test set.
        """
        if self.val_data_path is None:
            return {"c2st_xgb_val": float("nan")}, {}

        info = deepcopy(self.info)
        val_data = pd.read_csv(self.val_data_path)

        val_data.columns = range(len(val_data.columns))
        syn_data.columns = range(len(syn_data.columns))

        metadata = info['metadata']
        metadata['columns'] = {int(key): value for key, value in metadata['columns'].items()}

        new_val_data, new_syn_data, metadata = reorder(val_data, syn_data, info)

        score = XGBoostDetection.compute(
            real_data=new_val_data,       # <-- val instead of train
            synthetic_data=new_syn_data,
            metadata=metadata
        )

        out_metrics = {
            "c2st_xgb_val": score,
            "c2st_xgb_val_auc": 1 - score / 2,
        }
        out_extras = {}
        return out_metrics, out_extras  
    
    def plot_density(self, syn_data):
        syn_data_cp = deepcopy(syn_data)
        real_data = pd.read_csv(self.real_data_path)
        info = deepcopy(self.info)
        y_only = len(syn_data_cp.columns)==1
        if y_only:
            target_col_idx = info['target_col_idx'][0]
            target_col_name = info['column_names'][target_col_idx]
            syn_data_cp = self.complete_y_only_data(syn_data_cp, real_data, target_col_name)
        img = plot_density(syn_data_cp, real_data, info)
        return img
    
    def complete_y_only_data(self, syn_data, real_data, target_col_idx):
        syn_target_col = deepcopy(syn_data.iloc[:, 0])
        syn_data = deepcopy(real_data)
        syn_data[target_col_idx] = syn_target_col
        return syn_data
        
class XGBoostDetection(ScikitLearnClassifierDetectionMetric):
    name = "XGBoost Detection"

    @staticmethod
    def _get_classifier():
        return XGBClassifier(
            eval_metric="logloss",
            enable_categorical=True,
            tree_method="hist",
            random_state=0,
        )

def reorder(real_data, syn_data, info):
    num_col_idx = deepcopy(info['num_col_idx']) # BUG: info will be modified by += in the next few lines
    cat_col_idx = deepcopy(info['cat_col_idx'])
    target_col_idx = deepcopy(info['target_col_idx'])

    task_type = info['task_type']
    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    real_num_data = real_data[num_col_idx]
    real_cat_data = real_data[cat_col_idx]

    new_real_data = pd.concat([real_num_data, real_cat_data], axis=1)
    new_real_data.columns = range(len(new_real_data.columns))

    syn_num_data = syn_data[num_col_idx]
    syn_cat_data = syn_data[cat_col_idx]
    
    new_syn_data = pd.concat([syn_num_data, syn_cat_data], axis=1)
    new_syn_data.columns = range(len(new_syn_data.columns))

    
    metadata = info['metadata']

    columns = metadata['columns']
    metadata['columns'] = {}

    inverse_idx_mapping = info['inverse_idx_mapping']


    for i in range(len(new_real_data.columns)):
        if i < len(num_col_idx):
            metadata['columns'][i] = columns[num_col_idx[i]]
        else:
            metadata['columns'][i] = columns[cat_col_idx[i-len(num_col_idx)]]
    

    return new_real_data, new_syn_data, metadata
