CORE_CELLS = [
    "llama_3b",
    "llama_8b",
    "qwen_7b",
    "qwen_14b",
    "mistral_7b",
    "mistral_12b",
]

MODEL_CELLS = {
    "llama_3b": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "family": "Llama",
        "size": "3B",
        "writer_layers": list(range(14, 18)),
    },
    "llama_8b": {
        "model_id": "NousResearch/Hermes-3-Llama-3.1-8B",
        "family": "Llama",
        "size": "8B",
        "writer_layers": list(range(14, 25)),
    },
    "qwen_7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "family": "Qwen",
        "size": "7B",
        "writer_layers": list(range(24, 28)),
    },
    "qwen_14b": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "family": "Qwen",
        "size": "14B",
        "writer_layers": list(range(32, 48)),
    },
    "mistral_7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "family": "Mistral",
        "size": "7B",
        "writer_layers": list(range(19, 23)),
    },
    "mistral_12b": {
        "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "family": "Mistral",
        "size": "12B",
        "writer_layers": list(range(19, 25)),
    },
}
