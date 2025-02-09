from transformers import TextStreamer
from unsloth import FastLanguageModel
import torch

def load_model(model_path):
    """Loads a fine-tuned model from the given path."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    return model, tokenizer

def original_output(model, tokenizer, post, alpaca_prompt):
    """Generates output using the base model before fine-tuning."""
    FastLanguageModel.for_inference(model)
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Generate a funny response", 
                post,  
                "" 
            )
        ], return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    return tokenizer.batch_decode(outputs)

def fine_tuned_output(model, tokenizer, post, alpaca_prompt, streamer=False):
    """Generates output using the fine-tuned model."""
    FastLanguageModel.for_inference(model)
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Generate a funny response", 
                post, 
                "" 
            )
        ], return_tensors="pt").to("cuda")

    if streamer:
        text_streamer = TextStreamer(tokenizer)
        _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)
    else:
        outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
        return tokenizer.batch_decode(outputs)
