from llm2vec import LLM2Vec
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

class EmbedModel:
    def __init__(self, max_length):
        self.l2v = None

        tokenizer = AutoTokenizer.from_pretrained(
            "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"
        )
        config = AutoConfig.from_pretrained(
            "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
            trust_remote_code=True,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        model = PeftModel.from_pretrained(
            model,
            "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
        )
        model = model.merge_and_unload()  # This can take several minutes on cpu

        # Loading supervised model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
        model = PeftModel.from_pretrained(
            model, "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised"
        )

        # Wrapper for encoding and pooling operations
        self.l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=max_length)

    def encode(self, documents):
        reps = self.l2v.encode(documents)
        return reps