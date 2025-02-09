from install_datasets import install_datasets
from pre_process import humor_chain_formatter, sarcasm_humor_formatter
from train import train_model
from evaluate import load_model, original_output, fine_tuned_output
from utils import load_pretrained_model, configure_lora

def main():
    """Main function to run all steps sequentially."""

    # Step 1: Install datasets
    print("Installing datasets...")
    install_datasets()

    # Step 2: Preprocess datasets
    print("Processing Humor Chains dataset...")
    humor_dataset = humor_chain_formatter()

    print("Processing Sarcasm Humor dataset...")
    sarcasm_dataset = sarcasm_humor_formatter()

    # Step 3: Load pre-trained model
    print("Loading pre-trained model...")
    model, tokenizer = load_pretrained_model()

    # Step 4: Configure LoRA
    print("Configuring LoRA...")
    model = configure_lora(model)

    # Step 5: Fine-tune model
    print("Fine-tuning model...")
    train_model(humor_dataset)  

    # Step 6: Evaluate Model
    print("Evaluating model...")
    alpaca_prompt = "..."
    print("Original Output:", original_output(model, tokenizer, "Digging tunnels is a boring job.", alpaca_prompt))
    print("Fine-tuned Output:", fine_tuned_output(model, tokenizer, "Digging tunnels is a boring job.", alpaca_prompt))

if __name__ == "__main__":
    main()
