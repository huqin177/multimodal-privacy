from datasets import load_dataset

hf_path = "mattmdjaga/text-anonymization-benchmark-val-test"
save_path = "./data/court/text-anonymization-val-test.arrow"

# Load and save the dataset
ds = load_dataset(hf_path)
ds.save_to_disk(save_path)
