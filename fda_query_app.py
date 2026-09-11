import os
import streamlit as st
from dotenv import load_dotenv
from fastembed import TextEmbedding
from rag_pipe import load_vector_store, retrieve_context, ask_groq, EMBED_MODEL

load_dotenv()

st.set_page_config(page_title="FDA Docs Q&A", layout="centered")
st.title("📄 Ask your FDA documents")

@st.cache_resource
def get_embedder():
    return TextEmbedding(model_name=EMBED_MODEL)

@st.cache_resource
def get_collection():
    return load_vector_store()

embedder = get_embedder()
collection = get_collection()

st.caption(f"{collection.count()} chunks indexed.")

query = st.text_area(
    "Your question",
    height=100,
    placeholder="e.g. Briefly describe adaptive approaches",
)

if st.button("Ask") and query:
    with st.spinner("Searching documents and generating answer..."):
        docs, sources = retrieve_context(collection, embedder, query)
        answer = ask_groq(query, docs, sources)

    st.markdown("### Answer")
    st.write(answer)

    with st.expander("Sources used"):
        for src in set(sources):
            st.write(f"- {src}")