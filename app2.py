import os
import streamlit as st
from dotenv import load_dotenv
from fastembed import TextEmbedding
from rag_pipeline import build_vector_store, load_vector_store, retrieve_context, ask_groq, EMBED_MODEL

load_dotenv()

st.set_page_config(page_title="PDF Q&A Agent", layout="wide")
st.title("📄 PDF Q&A Agent")

@st.cache_resource
def get_embedder():
    return TextEmbedding(model_name=EMBED_MODEL)

@st.cache_resource
def get_collection():
    return load_vector_store()

embedder = get_embedder()

with st.sidebar:
    st.subheader("Index management")
    if st.button("Re-index PDFs (run after adding new files to FDA_Docs)"):
        with st.spinner("Indexing PDFs..."):
            build_vector_store()
        st.success("Re-indexed. Reload the page to refresh the collection.")

collection = get_collection()

query = st.text_area("Ask a question about your PDFs", height=100)

if st.button("Get answer") and query:
    with st.spinner("Retrieving context and generating answer..."):
        docs, sources = retrieve_context(collection, embedder, query)
        answer = ask_groq(query, docs, sources)

    st.markdown("### Answer")
    st.write(answer)

    with st.expander("Sources used"):
        for src in set(sources):
            st.write(f"- {src}")