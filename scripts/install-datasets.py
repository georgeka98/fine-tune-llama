# !kaggle datasets download -d sherinclaudia/sarcastic-comments-on-reddit

import zipfile

def install_datasets():
    """Extracts the sarcastic-comments dataset."""
    with zipfile.ZipFile("sarcastic-comments-on-reddit.zip", 'r') as zip_ref:
        zip_ref.extractall("data")  
