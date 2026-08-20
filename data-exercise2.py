from datasets import load_dataset
import time

dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

start = time.time()
count = 0

for example in dataset:
    count += 1
    if time.time() - start >= 10:
        break

print(f"Processed {count} examples in 10 seconds")
