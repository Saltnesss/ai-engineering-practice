from datasets import load_dataset

dataset = load_dataset("nyu-mll/glue" , "mrpc" , split = "train")

for i in range(5):
	print(dataset[i])
