import os
from dotenv import load_dotenv
from fastembed import TextEmbedding
from rag_pipe import load_vector_store, retrieve_context, ask_groq, EMBED_MODEL

load_dotenv()

def main():
    print("Loading embedder and vector store...")
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    collection = load_vector_store()
    print(f"Ready. {collection.count()} chunks indexed.\n")

    while True:
        query = input("Ask a question about your PDFs (or 'quit'): ").strip()
        if query.lower() in ("quit", "exit", ""):
            break

        docs, sources = retrieve_context(collection, embedder, query)
        answer = ask_groq(query, docs, sources)

        print("\n--- Answer ---")
        print(answer)
        print("\n--- Sources ---")
        print(set(sources))
        print()

if __name__ == "__main__":
    main()