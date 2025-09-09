# utils/preprocessing.py

import os
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Directory to save/load models locally
MODEL_DIR = os.path.join("models_cache")

os.makedirs(MODEL_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Lowercase
    - Remove HTML tags
    - Remove extra spaces
    - Remove non-alphanumeric chars (except punctuation)
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)  # remove HTML
    text = re.sub(r"[^a-z0-9.,!?;:()\[\]\s]", " ", text)  # keep only safe chars
    text = re.sub(r"\s+", " ", text).strip()

    return text


def download_model(model_name: str, save_dir: str):
    """
    Downloads a Hugging Face model and tokenizer locally (if not already present).
    """
    if save_dir is None:
        save_dir = os.path.join(MODEL_DIR, model_name.replace("/", "_"))

    if not os.path.exists(save_dir):
        print(f"Downloading {model_name} to {save_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
    else:
        print(f"Loading {model_name} from local cache {save_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        model = AutoModelForSequenceClassification.from_pretrained(save_dir)

    return tokenizer, model
