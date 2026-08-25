# make_val_split.py — run once per dataset you want to tune
import pandas as pd
from sklearn.model_selection import train_test_split
import sys, os

dataname = sys.argv[1]
train_path = f"synthetic/{dataname}/real.csv"   # this is the training data
df = pd.read_csv(train_path)

train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

# keep a held-out val for selection
val_df.to_csv(f"synthetic/{dataname}/val.csv", index=False)
print(f"{dataname}: train={len(train_df)}, val={len(val_df)} -> wrote synthetic/{dataname}/val.csv")