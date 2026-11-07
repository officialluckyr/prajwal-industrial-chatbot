
import numpy as np
import faiss
from fastembed import TextEmbedding

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Load the model once and reuse it
_model = None

def get_model():
    global _model

    if _model is None:
        print("Loading FastEmbed model...")
        _model = TextEmbedding(model_name=EMBEDDING_MODEL)
        print("Embedding model is ready.")

    return _model


def generate_embeddings(texts):
    model = get_model()

    embeddings = list(model.embed(texts))
    vectors = np.asarray(embeddings, dtype=np.float32)

    faiss.normalize_L2(vectors)

    return vectors


def build_vector_index(documents):
    if not documents:
        raise ValueError("No documents available.")

    texts = [document["text"] for document in documents]

    embeddings = generate_embeddings(texts)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_bytes = faiss.serialize_index(index).tobytes()

    return {
        "documents": documents,
        "index_bytes": index_bytes,
        "total_vectors": index.ntotal,
        "dimension": dimension
    }


def search_documents(query, knowledge_base, top_k=4):
    if not knowledge_base:
        return []

    query_embedding = generate_embeddings([query])

    query_embedding = np.ascontiguousarray(
        query_embedding,
        dtype=np.float32
    )

    index_bytes = np.frombuffer(
        knowledge_base["index_bytes"],
        dtype=np.uint8
    )

    index = faiss.deserialize_index(index_bytes)

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    documents = knowledge_base["documents"]

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        document = documents[int(idx)]

        results.append({
            "text": document["text"],
            "source": document["source"],
            "page": document["page"],
            "score": float(score)
        })

    return results
