import torch
import traceback
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    AutoModelForQuestionAnswering,
    PreTrainedTokenizerFast
)
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import os
import torch
import traceback
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer # <--- IMPORT THIS
from config import EMBEDDING_MODEL_NAME # etc.
from config import QA_MODEL_NAME, SUM_MODEL_NAME, EMBEDDING_MODEL_NAME, CROSS_ENCODER_MODEL_NAME

def load_qa_model():
    """Loads the Question Answering model and tokenizer."""
    print(f"Attempting to load Question Answering model: {QA_MODEL_NAME}...")
    pipeline_instance = None
    tokenizer_instance = None
    device_name = "CPU"
    device_id = -1
    pipeline_device = "cpu"

    try:
        if torch.cuda.is_available():
            device_id = 0
            pipeline_device = f"cuda:{device_id}"
            device_name = f"GPU {device_id} ({pipeline_device})"
            print(f"CUDA available for QA. Setting device to: {device_name}")
        else:
            print("CUDA not available for QA. Setting device to: CPU")

        tokenizer_instance = AutoTokenizer.from_pretrained(QA_MODEL_NAME, use_fast=True)
        model_instance = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
        pipeline_instance = pipeline(
            "question-answering",
            model=model_instance,
            tokenizer=tokenizer_instance,
            device=device_id # Use device_id for pipeline
        )
        print(f"QA model '{QA_MODEL_NAME}' loaded successfully on {device_name}.")

        if pipeline_instance:
            actual_device = pipeline_instance.device
            print(f"QA pipeline confirms usage of device: {actual_device}")
            # Update device name based on actual device used by pipeline
            if isinstance(actual_device, torch.device):
                 device_name = f"GPU {actual_device.index}" if actual_device.type == 'cuda' else "CPU"
            model_max_len = tokenizer_instance.model_max_length
            print(f"QA model ('{QA_MODEL_NAME}') max sequence length: {model_max_len}")
        if not isinstance(tokenizer_instance, PreTrainedTokenizerFast):
            print("WARNING: Loaded QA tokenizer is not a 'Fast' implementation.")

    except Exception as e:
        print(f"FATAL: Error loading QA model '{QA_MODEL_NAME}': {e}")
        traceback.print_exc()
        pipeline_instance = None
        tokenizer_instance = None
        device_name = "Not Initialized"

    return pipeline_instance, tokenizer_instance, device_name

def load_summarization_model():
    """Loads the Summarization model and tokenizer."""
    print(f"Attempting to load summarization model: {SUM_MODEL_NAME}...")
    pipeline_instance = None
    device_name = "CPU"
    device_id = -1

    try:
        if torch.cuda.is_available():
            device_id = 0
            device_name = f"GPU {device_id} (cuda:{device_id})"
            print(f"CUDA available for Summarizer. Setting device to: {device_name}")
        else:
            print("CUDA not available for Summarizer. Setting device to: CPU")

        tokenizer_instance = AutoTokenizer.from_pretrained(SUM_MODEL_NAME)
        model_instance = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL_NAME)
        pipeline_instance = pipeline(
            "summarization",
            model=model_instance,
            tokenizer=tokenizer_instance,
            device=device_id
        )
        print(f"Summarization model '{SUM_MODEL_NAME}' loaded successfully on {device_name}.")

        if pipeline_instance:
             actual_device = pipeline_instance.device
             print(f"Summarizer pipeline confirms usage of device: {actual_device}")
             if isinstance(actual_device, torch.device):
                  device_name = f"GPU {actual_device.index}" if actual_device.type == 'cuda' else "CPU"

    except Exception as e:
        print(f"FATAL: Error loading summarization model '{SUM_MODEL_NAME}': {e}")
        traceback.print_exc()
        pipeline_instance = None
        device_name = "Not Initialized"

    return pipeline_instance, device_name

# model_loader.py

def load_embedding_model():
    """Loads the Sentence Embedding model AND its tokenizer.""" # Updated docstring
    print(f"Attempting to load Sentence Embedding model: {EMBEDDING_MODEL_NAME}...")
    model_instance = None
    tokenizer_instance = None # <--- INITIALIZE TOKENIZER VARIABLE
    device_name = "CPU"
    device = "cpu"

    try:
        # Determine Device (using your existing logic)
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device() # Use current_device for flexibility
            device = f"cuda:{device_id}"
            device_name = f"GPU {device_id}" # Simpler name is fine too
            print(f"CUDA available for Embedding Model. Setting device to: {device_name} ({device})")
        else:
            print("CUDA not available for Embedding Model. Setting device to: CPU")

        # Load the Sentence Transformer model
        model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        print(f"Sentence Embedding model '{EMBEDDING_MODEL_NAME}' loaded successfully on {device_name}.")

        # --- Load the Tokenizer --- # <--- ADD THIS BLOCK ---
        try:
            tokenizer_instance = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
            print(f"Associated Tokenizer for '{EMBEDDING_MODEL_NAME}' loaded successfully.")
        except Exception as E_tok:
            print(f"Warning: Could not automatically load tokenizer for '{EMBEDDING_MODEL_NAME}'. Text splitting might use fallback. Error: {E_tok}")
            # tokenizer_instance remains None
        # --- End Tokenizer Loading --- #

    except Exception as e:
        print(f"FATAL: Error loading Sentence Embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        traceback.print_exc()
        model_instance = None
        tokenizer_instance = None # Ensure tokenizer is None on model load failure too
        device_name = "Failed to Load" # More descriptive status
        device = "cpu" # Fallback

    # --- Return all four values --- #
    return model_instance, tokenizer_instance, device, device_name
def load_cross_encoder_model():
    """Loads the Cross-Encoder model."""
    print(f"Attempting to load Cross-Encoder model: {CROSS_ENCODER_MODEL_NAME}...")
    model_instance = None
    device_name = "CPU"
    device = "cpu"

    try:
        if torch.cuda.is_available():
            device_id = 0
            device = f"cuda:{device_id}"
            device_name = f"GPU {device_id} ({device})"
            print(f"CUDA available for Cross-Encoder Model. Setting device to: {device_name}")
        else:
            print("CUDA not available for Cross-Encoder Model. Setting device to: CPU")

        # Note: max_length is important for cross-encoders
        model_instance = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=device, max_length=512)
        print(f"Cross-Encoder model '{CROSS_ENCODER_MODEL_NAME}' loaded successfully on {device_name}.")
    except Exception as e:
        print(f"FATAL: Error loading Cross-Encoder model '{CROSS_ENCODER_MODEL_NAME}': {e}")
        traceback.print_exc()
        model_instance = None
        device_name = "Not Initialized"
        device = "cpu" # Fallback

    return model_instance, device, device_name