# download_dataset.py

from datasets import load_dataset

# Define the dataset name and the local folder to save it in
dataset_name = "cnn_dailymail"
dataset_version = "3.0.0"
local_path = "./local_cnn_dailymail" # The folder where it will be saved

print(f"Downloading '{dataset_name}' from the Hub...")
# Load the dataset from the Hugging Face Hub
raw_datasets = load_dataset(dataset_name, dataset_version)

print(f"Saving dataset to local disk at: {local_path}")
# Save the entire dataset object to the specified folder
raw_datasets.save_to_disk(local_path)

print("Download and save complete!")