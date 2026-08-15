import torch

def get_inference_engine(model_id: str):
    if torch.cuda.is_available():
        print("NVIDIA GPU detected. Initializing high-throughput vLLM engine...")
        from vllm import LLM
        return LLM(model=model_id)
    else:
        print("No GPU detected. Falling back to Hugging Face PyTorch CPU engine...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float32, 
            device_map="cpu"
        )
        return model, tokenizer