"""Dataset-agnostic table encoding for the released HARPOON baselines.

The column convention intentionally matches ``baselines/harpoon/dataset.py``:
all numerical columns first, followed by one-hot categorical columns.  The
original DataFrame column order is restored when samples are written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="error")
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(sparse=False, handle_unknown="error")


@dataclass
class BaselineTable:
    train: pd.DataFrame
    test: pd.DataFrame
    info: dict
    numerical_columns: list[str]
    categorical_columns: list[str]
    encoder: OneHotEncoder

    @property
    def model_columns(self) -> list[str]:
        return self.numerical_columns + self.categorical_columns

    @property
    def numerical_count(self) -> int:
        return len(self.numerical_columns)

    @property
    def encoded_dim(self) -> int:
        return self.numerical_count + (
            sum(len(x) for x in self.encoder.categories_)
            if self.categorical_columns
            else 0
        )

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        nums = frame[self.numerical_columns].to_numpy(dtype=np.float64)
        if not self.categorical_columns:
            return nums
        cats = self.encoder.transform(
            frame[self.categorical_columns].astype(str)
        )
        return np.concatenate((nums, cats), axis=1)

    def decode(self, encoded: np.ndarray) -> pd.DataFrame:
        encoded = np.asarray(encoded)
        data: dict[str, object] = {}
        for index, column in enumerate(self.numerical_columns):
            data[column] = encoded[:, index]
        if self.categorical_columns:
            cats = self.encoder.inverse_transform(encoded[:, self.numerical_count :])
            for index, column in enumerate(self.categorical_columns):
                data[column] = cats[:, index]
        modeled = pd.DataFrame(data)
        return modeled.loc[:, self.train.columns]

    def standardization(self) -> tuple[np.ndarray, np.ndarray]:
        encoded = self.encode(self.train)
        n_num = self.numerical_count
        mean = np.concatenate(
            (encoded[:, :n_num].mean(axis=0), np.zeros(encoded.shape[1] - n_num))
        )
        std = np.concatenate(
            (encoded[:, :n_num].std(axis=0), np.ones(encoded.shape[1] - n_num))
        )
        std[~np.isfinite(std) | (std == 0)] = 1.0
        return mean, std

    def categorical_slices(self) -> dict[str, slice]:
        output = {}
        offset = self.numerical_count
        if not self.categorical_columns:
            return output
        for column, categories in zip(self.categorical_columns, self.encoder.categories_):
            output[column] = slice(offset, offset + len(categories))
            offset += len(categories)
        return output


def load_baseline_table(
    *,
    dataname: str | None = None,
    train_data: str | Path | None = None,
    test_data: str | Path | None = None,
    info_file: str | Path | None = None,
) -> BaselineTable:
    """Load any TabDiff-format dataset from explicit paths or ``data/NAME``."""
    if dataname is None and any(x is None for x in (train_data, test_data, info_file)):
        raise ValueError("give --dataname or all of train/test/info paths")
    train_path = Path(train_data or f"data/{dataname}/train.csv")
    test_path = Path(test_data or f"data/{dataname}/test.csv")
    info_path = Path(info_file or f"data/{dataname}/info.json")
    for path in (train_path, test_path, info_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if list(train.columns) != list(test.columns):
        raise ValueError("train and test columns differ")
    with info_path.open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    numerical = list(info["num_col_idx"])
    categorical = list(info["cat_col_idx"])
    target = list(info["target_col_idx"])
    if info["task_type"] == "regression":
        numerical = target + numerical
    else:
        categorical = target + categorical
    if sorted(numerical + categorical) != list(range(len(train.columns))):
        raise ValueError("info.json indices must cover every table column exactly once")
    numerical_columns = [str(train.columns[index]) for index in numerical]
    categorical_columns = [str(train.columns[index]) for index in categorical]
    encoder = _one_hot_encoder()
    if categorical_columns:
        # This is the released HARPOON preprocessing convention: schema is fit
        # on train+test, while model parameters are fit on train only.
        schema = pd.concat((train, test), ignore_index=True)
        encoder.fit(schema[categorical_columns].astype(str))
    return BaselineTable(
        train=train,
        test=test,
        info=info,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        encoder=encoder,
    )


def query_categorical_observations(
    table: BaselineTable,
    query: dict,
    num_rows: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Draw exact observations for categorical ``in`` predicates.

    HARPOON's GReaT/DiffPuter baselines accept observed cells, not sets.  For a
    set predicate we draw from the empirical training marginal restricted to
    the allowed set.  Numeric range predicates deliberately remain missing.
    """
    observed = pd.DataFrame(
        np.nan,
        index=np.arange(num_rows),
        columns=table.model_columns,
        dtype=object,
    )
    decisions = []
    categorical_names = set(table.categorical_columns)
    numerical_names = set(table.numerical_columns)
    for predicate in query.get("predicates", []):
        column = str(predicate["col"])
        modality = predicate.get("modality")
        operation = predicate.get("op")
        if modality == "numeric" and operation == "between":
            if column not in numerical_names:
                raise ValueError(f"query numeric column {column!r} is not numerical")
            continue
        if modality != "categorical" or operation != "in":
            raise ValueError(f"unsupported predicate for baseline adapter: {predicate}")
        if column not in categorical_names:
            raise ValueError(f"query categorical column {column!r} is not categorical")
        allowed = {str(value) for value in predicate["values"]}
        training = table.train[column].astype(str)
        counts = training[training.isin(allowed)].value_counts()
        if counts.empty:
            raise ValueError(
                f"categorical predicate {column!r} has no allowed value in training data"
            )
        values = counts.index.to_numpy(dtype=object)
        probabilities = (counts / counts.sum()).to_numpy(dtype=np.float64)
        observed[column] = rng.choice(values, size=num_rows, p=probabilities)
        decisions.append(
            {
                "column": column,
                "allowed_values": sorted(allowed),
                "available_values": values.tolist(),
                "training_probabilities": probabilities.tolist(),
            }
        )
    return observed, {
        "categorical_set_adapter": "training_marginal_restricted_to_allowed_set",
        "numerical_interval_adapter": "left_missing_unconditional",
        "categorical_decisions": decisions,
    }
