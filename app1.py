import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

st.set_page_config(page_title="Multi-Tab Demo", layout="wide")

# ---------------------------------------------------------------
# Cache the model so it only loads once across reruns/tabs
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
    )
    llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
    return llm

def ask_model(llm, query, max_tokens=300):
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful, knowledgeable assistant."},
            {"role": "user", "content": query},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response["choices"][0]["message"]["content"]

# ---------------------------------------------------------------
# App layout
# ---------------------------------------------------------------
st.title("Multi-Tab Streamlit Demo")

tab1, tab2, tab3 = st.tabs(["🤖 Ask the Model", "🔢 Live Calculator", "📊 Text Stats"])

# --- Tab 1: HF model Q&A ---
with tab1:
    st.subheader("Ask an open-ended question")
    query = st.text_area(
        "Your question",
        value="Explain a crossover design clinical study",
        height=100,
        key="model_query",
    )
    max_tokens = st.slider("Max response length (tokens)", 50, 500, 300, key="model_tokens")

    if st.button("Generate answer", key="model_button"):
        with st.spinner("Loading model / generating..."):
            llm = load_model()
            answer = ask_model(llm, query, max_tokens)
        st.session_state["model_answer"] = answer

    if "model_answer" in st.session_state:
        st.markdown("**Answer:**")
        st.write(st.session_state["model_answer"])

# --- Tab 2: simple reactive input -> output (no button needed) ---
with tab2:
    st.subheader("Simple live-updating calculator")
    a = st.number_input("Value A", value=10.0, key="calc_a")
    b = st.number_input("Value B", value=5.0, key="calc_b")
    op = st.selectbox("Operation", ["+", "-", "*", "/"], key="calc_op")

    ops = {"+": a + b, "-": a - b, "*": a * b, "/": (a / b if b != 0 else float("nan"))}
    st.metric("Result", ops[op])

# --- Tab 3: text input -> stats, updates as you type ---
with tab3:
    st.subheader("Live text statistics")
    text = st.text_area("Type or paste text", "Type something here...", key="stats_text")

    words = text.split()
    st.write(f"**Word count:** {len(words)}")
    st.write(f"**Character count:** {len(text)}")
    st.write(f"**Sentence count (approx):** {text.count('.') + text.count('!') + text.count('?')}")