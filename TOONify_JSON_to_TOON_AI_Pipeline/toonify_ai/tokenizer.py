# Token Estimation

import tiktoken # Fast BPE (Byte Pair Encoding) tokenizer developed by OpenAI.

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("o200k_base") # Defaulting to the most modern standard if model is unknown.

    return len(enc.encode(text))