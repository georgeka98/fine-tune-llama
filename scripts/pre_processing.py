# from datasets import load_dataset
# import pandas as pd

# def humor_chain_formatter():
    
#     alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

#     ### Instruction:
#     {}

#     ### Input:
#     {}

#     ### Response:
#     {}"""

#     EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
#     def formatting_prompts_func(examples):
#         conversations = examples["conversations"]
#         texts = []
#         for conversation in conversations:
#             # Must add EOS_TOKEN, otherwise your generation will go on forever!
#             input_text = conversation[0]["value"]
#             output_text = conversation[1]["value"]

#             text = alpaca_prompt.format("Generate a funny response", input_text, output_text) + EOS_TOKEN
#             texts.append(text)
#         return { "text" : texts, }
#     pass

#     dataset = load_dataset("ZSvedic/humor-chains", split = "train")
#     dataset = dataset.map(formatting_prompts_func, batched = True,)



# def sarcasm_humor_formatter():
    
#     # Load the dataset
#     df = pd.read_csv("data/train-balanced-sarcasm.csv")

#     # Display first few rows
#     print(df.head())
#     print(df.columns)

#     # Select the relevant text column
#     df = df.rename(columns={"comment": "text"})  # Ensure the key is "text" for fine-tuning


#     # Remove any NaN values (if present)
#     df = df.dropna(subset=["text", "ups", "downs"])  # Ensure "ups" and "downs" are not NaN

#     # **Filter the dataset based on 'ups' and 'downs' values**
#     df = df[(df["ups"] > 50) & (df["downs"] > -10)]

#     # Convert to Hugging Face dataset
#     dataset = Dataset.from_pandas(df)

#     # Split dataset (90% train, 10% validation)
#     dataset = dataset.train_test_split(test_size=0.1)

#     # Alpaca-style prompt template
#     alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

#     ### Instruction:
#     {}

#     ### Input:
#     {}

#     ### Response:
#     {}"""

#     EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

#     def formatting_prompts_func(examples):
#         texts = []

#         # Extract subreddit as input and parent_comment as output
#         for sub, parent in zip(examples["subreddit"], examples["parent_comment"]):
#             input_text = sub  # Use the subreddit name as input
#             output_text = parent  # Use the parent_comment as output

#             # Format using Alpaca-style prompt
#             text = alpaca_prompt.format("Generate a funny response", input_text, output_text) + EOS_TOKEN
#             texts.append(text)

#         return {"text": texts}

#     # Apply the function to the filtered dataset
#     dataset = dataset["train"].map(formatting_prompts_func, batched=True)

#     num_rows = df.shape[0]
#     print(f"Number of rows in the dataset: {num_rows}")






from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer

def humor_chain_formatter():
    """Formats the Humor-Chains dataset into Alpaca-style prompts."""
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token 

    def formatting_prompts_func(examples):
        texts = []
        for conversation in examples["conversations"]:
            input_text = conversation[0]["value"]
            output_text = conversation[1]["value"]
            text = alpaca_prompt.format("Generate a funny response", input_text, output_text) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts }

    dataset = load_dataset("ZSvedic/humor-chains", split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset

def sarcasm_humor_formatter(tokenizer):
    """Processes the sarcasm dataset and formats it for fine-tuning."""
    df = pd.read_csv("data/train-balanced-sarcasm.csv")
    df = df.rename(columns={"comment": "text"}).dropna(subset=["text", "ups", "downs"])
    df = df[(df["ups"] > 50) & (df["downs"] > -10)]

    dataset = load_dataset("csv", data_files="data/train-balanced-sarcasm.csv")["train"]

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token  

    def formatting_prompts_func(examples):
        texts = []
        for sub, parent in zip(examples["subreddit"], examples["parent_comment"]):
            text = alpaca_prompt.format("Generate a funny response", sub, parent) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset
