from datasets import load_dataset
import os

dataset = load_dataset("nyu-mll/glue", "mrpc", split="train")

dataset.to_csv("mrpc_train.csv")
dataset.to_parquet("mrpc_train.parquet")

csv_size = os.path.getsize("mrpc_train.csv")
parquet_size = os.path.getsize("mrpc_train.parquet")

print(f"CSV size: {csv_size} bytes")
print(f"Parquet size: {parquet_size} bytes")
print(f"Parquet is {csv_size / parquet_size:.1f}x smaller than CSV")
