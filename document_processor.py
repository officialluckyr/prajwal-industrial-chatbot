import fitz
from pathlib import Path


def process_pdfs(
    file_paths,
    chunk_size=1000,
    overlap=150
):

    all_chunks = []

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

                all_chunks.append({

                    "text": chunk,

                    "source": filename,

                    "page": page_number

                })

        pdf.close()

    return all_chunks