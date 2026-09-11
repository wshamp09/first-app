import os
import glob
from dotenv import load_dotenv
from pypdf import PdfReader
from fastembed import TextEmbedding
import chromadb
from groq import Groq

load_dotenv()

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
PDF_DIR = os.path.expanduser("~/Documents/FDA_Docs")
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
GROQ_MODEL = "openai/gpt-oss-120b"

# ---------------------------------------------------------------
# PDF loading + chunking
# ---------------------------------------------------------------
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]

# ---------------------------------------------------------------
# Embedding + vector store
# ---------------------------------------------------------------
def build_vector_store():
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection("pdf_docs")

    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in '{PDF_DIR}/'. Add some and re-run.")
        return collection

    for path in pdf_files:
        print(f"Processing {path}...")
        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)
        if not chunks:
            continue
        embeddings = [emb.tolist() for emb in embedder.embed(chunks)]
        ids = [f"{os.path.basename(path)}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": os.path.basename(path), "chunk": i} for i in range(len(chunks))]

        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    print(f"Indexed {len(pdf_files)} PDF(s) into '{CHROMA_DIR}'.")
    return collection

def load_vector_store():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection("pdf_docs")

# ---------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------
def retrieve_context(collection, embedder, query, top_k=4):
    query_embedding = list(embedder.embed([query]))[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    docs = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return docs, sources

# ---------------------------------------------------------------
# Groq-powered answer generation
# ---------------------------------------------------------------
def ask_groq(query, context_chunks, sources):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    context_block = "\n\n---\n\n".join(
        f"[Source: {src}]\n{chunk}" for chunk, src in zip(context_chunks, sources)
    )

    prompt = f"""Answer the question using only the context below. If the context doesn't contain the answer, say so.

Context:
{context_block}

Question: {query}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content

# ---------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    collection = build_vector_store()
    embedder = TextEmbedding(model_name=EMBED_MODEL)

    query = input("\nAsk a question about your PDFs: ")
    docs, sources = retrieve_context(collection, embedder, query)
    answer = ask_groq(query, docs, sources)

    print("\n--- Answer ---")
    print(answer)
    print("\n--- Sources used ---")
    print(set(sources))