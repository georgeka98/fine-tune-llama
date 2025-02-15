import os
import zipfile
import kaggle

def download_dataset():
    """Downloads the sarcastic-comments dataset from Kaggle."""
    dataset = "sherinclaudia/sarcastic-comments-on-reddit"
    output_path = "sarcastic-comments-on-reddit.zip"

    if not os.path.exists(output_path):
        print("Downloading dataset from Kaggle...")
        os.system(f"kaggle datasets download -d {dataset} -o")
        print("Download complete.")

def install_datasets():
    """Extracts the sarcastic-comments dataset."""
    zip_path = "sarcastic-comments-on-reddit.zip"
    extract_dir = "data"

    if not os.path.exists(extract_dir):  # Avoid re-extraction
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction complete.")

# Run the functions before executing the rest of the script
download_dataset()
install_datasets()
