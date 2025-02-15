# Fine-Tuning LLaMA 3.1 with Unsloth

## 🚀 Overview
This repository provides scripts and instructions for fine-tuning **LLaMA 3.1** on humor and sarcasm datasets using **Unsloth** for optimized performance. The setup includes dataset preparation, fine-tuning, and evaluation.

---

## 📦 Installation
### 1️⃣ Create a Virtual Environment (Optional but Recommended)
To keep dependencies organized, it's recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate    # On Windows
```

### 2️⃣ Install Required Python Packages
Run the following command to install necessary dependencies:
```bash
pip install -r requirements.txt
```

If the `requirements.txt` file is missing, manually install the packages:
```bash
pip install unsloth kaggle zipfile
```

#### Install the latest Unsloth version:
```bash
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

---

## 📂 Uploading and Extracting Model Checkpoints
If you have pre-trained model checkpoints (e.g., `outputs.zip`), upload them and extract as follows:
```bash
unzip -t outputs.zip
```
Or extract manually in Python:
```python
import zipfile
with zipfile.ZipFile("outputs.zip", 'r') as zip_ref:
    zip_ref.extractall("models/fine_tuned/")
```

---

## 📥 Downloading Datasets
### 1️⃣ Install Kaggle CLI
To access Kaggle datasets, install the Kaggle API tool:
```bash
pip install kaggle
```

### 2️⃣ Authenticate with Kaggle
1. Go to [Kaggle](https://www.kaggle.com/), sign in, and navigate to **Account Settings**.
2. Scroll down to the **API** section and click "Create New API Token".
3. A file named `kaggle.json` will be downloaded.
4. Move this file to the correct location:
   ```bash
   mkdir -p ~/.kaggle
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 3️⃣ Download the Sarcastic Comments Dataset
```bash
kaggle datasets download -d sherinclaudia/sarcastic-comments-on-reddit
```

### 4️⃣ Extract the Dataset
```python
import zipfile
with zipfile.ZipFile("sarcastic-comments-on-reddit.zip", 'r') as zip_ref:
    zip_ref.extractall("data")  # Extract to the "data" folder
```

---

## 🔧 Running the Fine-Tuning Pipeline
Once everything is set up, run the entire pipeline by executing:
```bash
python main.py
```
This will:
1. Install and extract datasets
2. Pre-process datasets
3. Load and configure LLaMA 3.1
4. Fine-tune the model
5. Evaluate performance

---

## 📌 Notes
- Ensure you have a GPU-enabled environment for faster fine-tuning.
- Adjust the batch size and number of epochs in `train.py` based on available resources.
- Use `screen` or `tmux` when running on a remote server to avoid interruptions.

---

## 🎯 Conclusion
You're all set to fine-tune LLaMA 3.1 using Unsloth! 🚀 If you have any issues, feel free to open an issue or modify the scripts accordingly. Happy training! 🎉



conda activate /afs/inf.ed.ac.uk/group/project/diss-project/conda-envs/llama-env