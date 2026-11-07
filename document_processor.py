import fitz
from pathlib import Path
from transformers import AutoTokenizer

HF_TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"
_tokenizer = None


def get_hf_tokenizer():
    """Load the Hugging Face tokenizer once for token-aware metadata."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_NAME)
    return _tokenizer


def process_pdfs(
    file_paths,
    chunk_size=1000,
    overlap=150
):
    all_chunks = []
    tokenizer = get_hf_tokenizer()

    for file_path in file_paths:
        pdf = fitz.open(file_path)
        filename = Path(file_path).name

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            text = " ".join(text.split())

            if not text.strip():
                continue

            step = chunk_size - overlap

            for start in range(0, len(text), step):
                chunk = text[start:start + chunk_size]

                if len(chunk.strip()) < 40:
                    continue

                token_ids = tokenizer.encode(chunk, add_special_tokens=False)

                all_chunks.append({
                    "text": chunk,
                    "source": filename,
                    "page": page_number,
                    "token_count": len(token_ids)
                })

        pdf.close()

    return all_chunks